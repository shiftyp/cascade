"""Prometheus metrics for CASCADE data collector.

Exposes metrics in Prometheus format for monitoring and alerting.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    REGISTRY,
)
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..models import SessionLocal, RecordingSession, KiwiSDRSource, WebSDRSource


# Collection metrics
recording_sessions_total = Counter(
    "cascade_recording_sessions_total",
    "Total number of recording sessions started",
    ["status", "band", "sdr_type"],
)

recording_duration_seconds = Histogram(
    "cascade_recording_duration_seconds",
    "Duration of recording sessions",
    ["band", "sdr_type"],
    buckets=[60, 360, 600, 1800, 3600, 7200],  # 1m, 6m, 10m, 30m, 1h, 2h
)

active_recordings = Gauge(
    "cascade_active_recordings",
    "Number of currently active recordings",
    ["band", "sdr_type"],
)

# SDR metrics
sdrs_available = Gauge(
    "cascade_sdrs_available",
    "Number of available SDRs",
    ["sdr_type"],
)

sdrs_in_use = Gauge(
    "cascade_sdrs_in_use",
    "Number of SDRs currently in use",
    ["sdr_type"],
)

sdr_daily_usage_minutes = Gauge(
    "cascade_sdr_daily_usage_minutes",
    "Daily usage in minutes per SDR",
    ["sdr_url", "sdr_type"],
)

# Data metrics
data_collected_bytes = Counter(
    "cascade_data_collected_bytes",
    "Total bytes of data collected",
    ["band", "sdr_type"],
)

data_uploaded_bytes = Counter(
    "cascade_data_uploaded_bytes",
    "Total bytes uploaded to Tigris",
    ["band"],
)

# Quality metrics
recording_quality_score = Histogram(
    "cascade_recording_quality_score",
    "Quality score distribution",
    ["band"],
    buckets=[0, 25, 50, 75, 90, 95, 100],
)

recording_failures = Counter(
    "cascade_recording_failures",
    "Number of failed recordings",
    ["reason", "sdr_type"],
)

# Geographic diversity metrics
geographic_diversity_index = Gauge(
    "cascade_geographic_diversity_index",
    "Simpson's diversity index for geographic coverage",
)

continental_coverage = Gauge(
    "cascade_continental_coverage",
    "Coverage by continent",
    ["continent"],
)

# Processing metrics
processing_queue_depth = Gauge(
    "cascade_processing_queue_depth",
    "Number of recordings awaiting processing",
)

processing_version = Gauge(
    "cascade_processing_version",
    "Current processing pipeline version",
)

processed_recordings = Counter(
    "cascade_processed_recordings_total",
    "Total recordings processed",
    ["status", "version"],
)


class MetricsCollector:
    """Collects and updates Prometheus metrics."""

    def __init__(self, update_interval: int = 30):
        """Initialize metrics collector.

        Args:
            update_interval: Seconds between metric updates
        """
        self.update_interval = update_interval
        self.running = False

    async def start(self):
        """Start metrics collection loop."""
        self.running = True
        while self.running:
            try:
                await self.update_metrics()
            except Exception as e:
                print(f"Error updating metrics: {e}")

            await asyncio.sleep(self.update_interval)

    async def stop(self):
        """Stop metrics collection."""
        self.running = False

    async def update_metrics(self):
        """Update all metrics from database."""
        db = SessionLocal()
        try:
            # Update recording metrics
            await self._update_recording_metrics(db)

            # Update SDR metrics
            await self._update_sdr_metrics(db)

            # Update quality metrics
            await self._update_quality_metrics(db)

            # Update processing metrics
            await self._update_processing_metrics(db)

        finally:
            db.close()

    async def _update_recording_metrics(self, db: Session):
        """Update recording-related metrics."""
        # Active recordings by status
        active = db.query(RecordingSession).filter(
            RecordingSession.status == "recording"
        ).all()

        # Reset gauge
        active_recordings._metrics.clear()

        for session in active:
            sdr_type = "websdr" if session.websdr_id else "kiwisdr"
            active_recordings.labels(
                band=session.band or "unknown",
                sdr_type=sdr_type,
            ).inc()

        # Total sessions by status
        status_counts = db.query(
            RecordingSession.status,
            func.count(RecordingSession.session_id)
        ).group_by(RecordingSession.status).all()

        for status, count in status_counts:
            recording_sessions_total.labels(
                status=status,
                band="all",
                sdr_type="all",
            )._value._value = count

    async def _update_sdr_metrics(self, db: Session):
        """Update SDR availability metrics."""
        # KiwiSDR metrics
        kiwisdrs = db.query(KiwiSDRSource).filter(
            KiwiSDRSource.active == True
        ).all()

        available_kiwis = sum(
            1 for sdr in kiwisdrs
            if sdr.remaining_daily_minutes > 10
        )

        sdrs_available.labels(sdr_type="kiwisdr").set(available_kiwis)

        # WebSDR metrics
        websdrs = db.query(WebSDRSource).filter(
            WebSDRSource.active == True
        ).all()

        available_websdrs = sum(
            1 for sdr in websdrs
            if not sdr.daily_limit_minutes or
            (sdr.daily_limit_minutes - sdr.daily_usage_minutes) > 10
        )

        sdrs_available.labels(sdr_type="websdr").set(available_websdrs)

        # Usage metrics
        for sdr in kiwisdrs:
            sdr_daily_usage_minutes.labels(
                sdr_url=sdr.url,
                sdr_type="kiwisdr"
            ).set(sdr.daily_usage_minutes)

        for sdr in websdrs:
            sdr_daily_usage_minutes.labels(
                sdr_url=sdr.url,
                sdr_type="websdr"
            ).set(sdr.daily_usage_minutes)

    async def _update_quality_metrics(self, db: Session):
        """Update quality metrics."""
        # Failed recordings in last hour
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)

        failed = db.query(RecordingSession).filter(
            RecordingSession.status == "failed",
            RecordingSession.created_at >= one_hour_ago
        ).all()

        # Categorize failures
        failure_reasons = {}
        for session in failed:
            reason = "unknown"
            if session.error_message:
                if "connect" in session.error_message.lower():
                    reason = "connection"
                elif "timeout" in session.error_message.lower():
                    reason = "timeout"
                elif "limit" in session.error_message.lower():
                    reason = "limit_reached"

            sdr_type = "websdr" if session.websdr_id else "kiwisdr"
            key = (reason, sdr_type)
            failure_reasons[key] = failure_reasons.get(key, 0) + 1

        # Update counter
        for (reason, sdr_type), count in failure_reasons.items():
            recording_failures.labels(
                reason=reason,
                sdr_type=sdr_type
            )._value._value = count

    async def _update_processing_metrics(self, db: Session):
        """Update processing metrics."""
        from ..config.processing_config import PROCESSING_VERSION

        # Current processing version
        processing_version.set(PROCESSING_VERSION)

        # Queue depth - unprocessed recordings
        unprocessed = db.query(func.count(RecordingSession.session_id)).filter(
            RecordingSession.processing_status == "unprocessed",
            RecordingSession.status == "completed"
        ).scalar()

        processing_queue_depth.set(unprocessed or 0)

        # Processed by version
        processed_counts = db.query(
            RecordingSession.processing_version,
            RecordingSession.processing_status,
            func.count(RecordingSession.session_id)
        ).group_by(
            RecordingSession.processing_version,
            RecordingSession.processing_status
        ).all()

        for version, status, count in processed_counts:
            if version is not None:
                processed_recordings.labels(
                    status=status,
                    version=str(version)
                )._value._value = count


def get_metrics() -> bytes:
    """Get metrics in Prometheus format.

    Returns:
        Metrics as bytes in Prometheus text format
    """
    return generate_latest(REGISTRY)