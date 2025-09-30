"""PostgreSQL metadata interface for recording sessions and analysis results.

Implements T035: PostgreSQL metadata interface.
"""

import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
from sqlalchemy.orm import sessionmaker
from sqlalchemy import and_, or_, func, desc

from ..models import (
    SessionLocal, RecordingSession, KiwiSDRSource, QRNSample,
    FT8Signal, WSPRSignal, PropagationRecord, SpaceWeatherData,
    CollectionSchedule, CollectionAlert
)

logger = logging.getLogger(__name__)


class MetadataDB:
    """Interface for PostgreSQL metadata operations."""

    def __init__(self, db_session=None):
        """Initialize metadata interface.

        Args:
            db_session: Optional database session (creates new if None)
        """
        self.db = db_session or SessionLocal()
        self.owns_session = db_session is None

    def create_recording_session(
        self,
        kiwisdr_url: str,
        center_frequency_hz: int,
        bandwidth_hz: int,
        start_time: datetime,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> RecordingSession:
        """Create new recording session.

        Args:
            kiwisdr_url: KiwiSDR URL
            center_frequency_hz: Center frequency
            bandwidth_hz: Bandwidth
            start_time: Start time
            correlation_id: Correlation ID for linked samples
            **kwargs: Additional session parameters

        Returns:
            Created RecordingSession
        """
        try:
            # Find KiwiSDR source
            kiwisdr = self.db.query(KiwiSDRSource).filter(
                KiwiSDRSource.url == kiwisdr_url
            ).first()

            if not kiwisdr:
                raise ValueError(f"KiwiSDR not found: {kiwisdr_url}")

            # Create session
            session = RecordingSession(
                kiwisdr_id=kiwisdr.kiwisdr_id,
                center_frequency_hz=center_frequency_hz,
                bandwidth_hz=bandwidth_hz,
                start_time=start_time,
                correlation_id=correlation_id,
                **kwargs
            )

            self.db.add(session)
            self.db.commit()
            self.db.refresh(session)

            logger.info(f"Created recording session: {session.session_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to create recording session: {e}")
            self.db.rollback()
            raise

    def complete_recording_session(
        self,
        session_id: str,
        end_time: datetime,
        file_path: str,
        file_size: int,
        quality_metrics: Dict[str, Any]
    ) -> RecordingSession:
        """Complete a recording session.

        Args:
            session_id: Session ID
            end_time: End time
            file_path: Path to stored file
            file_size: File size in bytes
            quality_metrics: Quality assessment metrics

        Returns:
            Updated RecordingSession
        """
        try:
            session = self.db.query(RecordingSession).filter(
                RecordingSession.session_id == session_id
            ).first()

            if not session:
                raise ValueError(f"Session not found: {session_id}")

            # Update session
            session.end_time = end_time
            session.iq_file_path = file_path
            session.file_size_bytes = file_size
            session.quality_score = quality_metrics.get('quality_score', 0.0)
            session.avg_noise_floor_dbm = quality_metrics.get('avg_noise_floor_dbm')
            session.signal_count = quality_metrics.get('signal_count', 0)
            session.processing_status = 'completed'

            self.db.commit()
            self.db.refresh(session)

            logger.info(f"Completed recording session: {session_id}")
            return session

        except Exception as e:
            logger.error(f"Failed to complete recording session: {e}")
            self.db.rollback()
            raise

    def store_ft8_signals(
        self, session_id: str, signals: List[Dict[str, Any]]
    ) -> int:
        """Store FT8 signals for a session.

        Args:
            session_id: Recording session ID
            signals: List of FT8 signal data

        Returns:
            Number of signals stored
        """
        try:
            stored_count = 0

            for signal_data in signals:
                ft8_signal = FT8Signal(
                    session_id=session_id,
                    **signal_data
                )
                self.db.add(ft8_signal)
                stored_count += 1

            self.db.commit()
            logger.info(f"Stored {stored_count} FT8 signals for session {session_id}")
            return stored_count

        except Exception as e:
            logger.error(f"Failed to store FT8 signals: {e}")
            self.db.rollback()
            return 0

    def store_wspr_signals(
        self, session_id: str, signals: List[Dict[str, Any]]
    ) -> int:
        """Store WSPR signals for a session.

        Args:
            session_id: Recording session ID
            signals: List of WSPR signal data

        Returns:
            Number of signals stored
        """
        try:
            stored_count = 0

            for signal_data in signals:
                wspr_signal = WSPRSignal(
                    session_id=session_id,
                    **signal_data
                )
                self.db.add(wspr_signal)
                stored_count += 1

            self.db.commit()
            logger.info(f"Stored {stored_count} WSPR signals for session {session_id}")
            return stored_count

        except Exception as e:
            logger.error(f"Failed to store WSPR signals: {e}")
            self.db.rollback()
            return 0

    def store_qrn_samples(
        self, session_id: str, qrn_data: List[Dict[str, Any]]
    ) -> int:
        """Store QRN analysis samples.

        Args:
            session_id: Recording session ID
            qrn_data: List of QRN sample data

        Returns:
            Number of samples stored
        """
        try:
            stored_count = 0

            for qrn_sample in qrn_data:
                qrn_record = QRNSample(
                    session_id=session_id,
                    **qrn_sample
                )
                self.db.add(qrn_record)
                stored_count += 1

            self.db.commit()
            logger.info(f"Stored {stored_count} QRN samples for session {session_id}")
            return stored_count

        except Exception as e:
            logger.error(f"Failed to store QRN samples: {e}")
            self.db.rollback()
            return 0

    def get_recording_sessions(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        band: Optional[str] = None,
        kiwisdr_url: Optional[str] = None,
        min_quality: float = 0.0,
        limit: int = 100
    ) -> List[RecordingSession]:
        """Query recording sessions with filters.

        Args:
            start_time: Filter by start time
            end_time: Filter by end time
            band: Filter by band
            kiwisdr_url: Filter by KiwiSDR URL
            min_quality: Minimum quality score
            limit: Maximum results

        Returns:
            List of matching sessions
        """
        query = self.db.query(RecordingSession)

        # Apply filters
        if start_time:
            query = query.filter(RecordingSession.start_time >= start_time)

        if end_time:
            query = query.filter(RecordingSession.start_time <= end_time)

        if band:
            query = query.filter(RecordingSession.band == band)

        if kiwisdr_url:
            kiwisdr = self.db.query(KiwiSDRSource).filter(
                KiwiSDRSource.url == kiwisdr_url
            ).first()
            if kiwisdr:
                query = query.filter(RecordingSession.kiwisdr_id == kiwisdr.kiwisdr_id)

        if min_quality > 0:
            query = query.filter(RecordingSession.quality_score >= min_quality)

        # Order by start time (newest first) and limit
        sessions = query.order_by(desc(RecordingSession.start_time)).limit(limit).all()

        return sessions

    def get_collection_statistics(
        self, days_back: int = 7
    ) -> Dict[str, Any]:
        """Get collection statistics.

        Args:
            days_back: Number of days to analyze

        Returns:
            Collection statistics
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days_back)

        try:
            # Basic session counts
            total_sessions = self.db.query(RecordingSession).count()
            recent_sessions = self.db.query(RecordingSession).filter(
                RecordingSession.start_time >= cutoff_time
            ).count()

            # Calculate total hours
            total_hours_query = self.db.query(
                func.sum(
                    func.extract('epoch', RecordingSession.end_time - RecordingSession.start_time) / 3600
                )
            ).filter(
                RecordingSession.end_time.isnot(None)
            ).scalar()

            total_hours = float(total_hours_query or 0)

            # Recent hours
            recent_hours_query = self.db.query(
                func.sum(
                    func.extract('epoch', RecordingSession.end_time - RecordingSession.start_time) / 3600
                )
            ).filter(
                RecordingSession.end_time.isnot(None),
                RecordingSession.start_time >= cutoff_time
            ).scalar()

            recent_hours = float(recent_hours_query or 0)

            # Quality metrics
            avg_quality = self.db.query(
                func.avg(RecordingSession.quality_score)
            ).filter(
                RecordingSession.start_time >= cutoff_time,
                RecordingSession.quality_score.isnot(None)
            ).scalar()

            # Active SDRs
            active_sdrs = self.db.query(
                func.count(func.distinct(RecordingSession.kiwisdr_id))
            ).filter(
                RecordingSession.start_time >= cutoff_time
            ).scalar()

            # Storage usage
            total_storage_query = self.db.query(
                func.sum(RecordingSession.file_size_bytes)
            ).filter(
                RecordingSession.file_size_bytes.isnot(None)
            ).scalar()

            total_storage = int(total_storage_query or 0)

            return {
                "period_days": days_back,
                "total_sessions": total_sessions,
                "recent_sessions": recent_sessions,
                "total_hours_collected": total_hours,
                "recent_hours_collected": recent_hours,
                "daily_rate_hours": recent_hours / days_back,
                "average_quality": float(avg_quality or 0),
                "active_sdrs": int(active_sdrs or 0),
                "total_storage_bytes": total_storage,
                "storage_tb": total_storage / (1024**4),
            }

        except Exception as e:
            logger.error(f"Error calculating collection statistics: {e}")
            return {"error": str(e)}

    def get_space_weather_correlation(
        self, session_id: str
    ) -> Optional[SpaceWeatherData]:
        """Get space weather data for a recording session.

        Args:
            session_id: Recording session ID

        Returns:
            Associated SpaceWeatherData or None
        """
        try:
            session = self.db.query(RecordingSession).filter(
                RecordingSession.session_id == session_id
            ).first()

            if not session:
                return None

            # Find nearest space weather observation
            space_weather = self.db.query(SpaceWeatherData).filter(
                SpaceWeatherData.observation_time <= session.start_time
            ).order_by(desc(SpaceWeatherData.observation_time)).first()

            return space_weather

        except Exception as e:
            logger.error(f"Error getting space weather correlation: {e}")
            return None

    def find_correlated_sessions(
        self, correlation_id: str
    ) -> List[RecordingSession]:
        """Find all sessions with the same correlation ID.

        Args:
            correlation_id: Correlation ID to search for

        Returns:
            List of correlated sessions
        """
        try:
            sessions = self.db.query(RecordingSession).filter(
                RecordingSession.correlation_id == correlation_id
            ).order_by(RecordingSession.start_time).all()

            return sessions

        except Exception as e:
            logger.error(f"Error finding correlated sessions: {e}")
            return []

    def get_propagation_summary(
        self, start_time: datetime, end_time: datetime
    ) -> Dict[str, Any]:
        """Get propagation analysis summary for time period.

        Args:
            start_time: Period start
            end_time: Period end

        Returns:
            Propagation summary
        """
        try:
            # Get sessions in time period
            sessions = self.db.query(RecordingSession).filter(
                RecordingSession.start_time >= start_time,
                RecordingSession.start_time <= end_time
            ).all()

            if not sessions:
                return {"error": "No sessions found in time period"}

            session_ids = [s.session_id for s in sessions]

            # Count signals by type
            ft8_count = self.db.query(FT8Signal).filter(
                FT8Signal.session_id.in_(session_ids)
            ).count()

            wspr_count = self.db.query(WSPRSignal).filter(
                WSPRSignal.session_id.in_(session_ids)
            ).count()

            # QRN analysis
            qrn_samples = self.db.query(QRNSample).filter(
                QRNSample.session_id.in_(session_ids)
            ).count()

            # Quality metrics
            avg_quality = self.db.query(
                func.avg(RecordingSession.quality_score)
            ).filter(
                RecordingSession.session_id.in_(session_ids)
            ).scalar()

            return {
                "period": {
                    "start": start_time.isoformat(),
                    "end": end_time.isoformat(),
                    "duration_hours": (end_time - start_time).total_seconds() / 3600
                },
                "sessions": {
                    "total_count": len(sessions),
                    "avg_quality": float(avg_quality or 0),
                },
                "signals": {
                    "ft8_count": ft8_count,
                    "wspr_count": wspr_count,
                    "total_signals": ft8_count + wspr_count
                },
                "qrn_analysis": {
                    "sample_count": qrn_samples,
                },
            }

        except Exception as e:
            logger.error(f"Error generating propagation summary: {e}")
            return {"error": str(e)}

    def cleanup_old_sessions(self, days_old: int = 90) -> int:
        """Clean up old session metadata (keep file references).

        Args:
            days_old: Delete sessions older than this many days

        Returns:
            Number of sessions cleaned up
        """
        cutoff_time = datetime.utcnow() - timedelta(days=days_old)

        try:
            # Only delete failed or unprocessed sessions
            old_sessions = self.db.query(RecordingSession).filter(
                RecordingSession.start_time < cutoff_time,
                or_(
                    RecordingSession.processing_status == 'failed',
                    and_(
                        RecordingSession.processing_status == 'pending',
                        RecordingSession.end_time.is_(None)
                    )
                )
            )

            count = old_sessions.count()
            old_sessions.delete()
            self.db.commit()

            logger.info(f"Cleaned up {count} old sessions")
            return count

        except Exception as e:
            logger.error(f"Error cleaning up old sessions: {e}")
            self.db.rollback()
            return 0

    def update_sdr_usage(
        self, kiwisdr_url: str, minutes_used: float
    ) -> bool:
        """Update SDR usage tracking.

        Args:
            kiwisdr_url: KiwiSDR URL
            minutes_used: Minutes of usage

        Returns:
            True if successful
        """
        try:
            sdr = self.db.query(KiwiSDRSource).filter(
                KiwiSDRSource.url == kiwisdr_url
            ).first()

            if not sdr:
                logger.warning(f"KiwiSDR not found for usage update: {kiwisdr_url}")
                return False

            # Update usage
            sdr.daily_usage_minutes += minutes_used
            sdr.total_usage_minutes += minutes_used
            sdr.last_connected = datetime.utcnow()

            self.db.commit()

            logger.debug(f"Updated usage for {kiwisdr_url}: +{minutes_used:.1f} min")
            return True

        except Exception as e:
            logger.error(f"Error updating SDR usage: {e}")
            self.db.rollback()
            return False

    def get_collection_health(self) -> Dict[str, Any]:
        """Get collection system health metrics.

        Returns:
            Health metrics
        """
        try:
            now = datetime.utcnow()
            last_hour = now - timedelta(hours=1)

            # Recent session activity
            recent_sessions = self.db.query(RecordingSession).filter(
                RecordingSession.start_time >= last_hour
            ).count()

            # Active SDRs
            active_sdrs = self.db.query(
                func.count(func.distinct(RecordingSession.kiwisdr_id))
            ).filter(
                RecordingSession.start_time >= last_hour
            ).scalar()

            # Failed sessions
            failed_sessions = self.db.query(RecordingSession).filter(
                RecordingSession.start_time >= last_hour,
                RecordingSession.processing_status == 'failed'
            ).count()

            # Average quality
            avg_quality = self.db.query(
                func.avg(RecordingSession.quality_score)
            ).filter(
                RecordingSession.start_time >= last_hour,
                RecordingSession.quality_score.isnot(None)
            ).scalar()

            # Space weather status
            latest_weather = self.db.query(SpaceWeatherData).order_by(
                desc(SpaceWeatherData.observation_time)
            ).first()

            health_score = 100
            if failed_sessions > recent_sessions * 0.1:  # >10% failure rate
                health_score -= 20
            if not recent_sessions:  # No recent activity
                health_score -= 30
            if not active_sdrs:  # No active SDRs
                health_score -= 30

            return {
                "timestamp": now.isoformat(),
                "health_score": max(0, health_score),
                "recent_activity": {
                    "sessions_last_hour": recent_sessions,
                    "active_sdrs": int(active_sdrs or 0),
                    "failed_sessions": failed_sessions,
                    "success_rate": (recent_sessions - failed_sessions) / max(recent_sessions, 1) * 100
                },
                "quality_metrics": {
                    "average_quality": float(avg_quality or 0),
                },
                "space_weather": {
                    "last_update": latest_weather.observation_time.isoformat() if latest_weather else None,
                    "k_index": latest_weather.k_index if latest_weather else None,
                    "solar_flux": latest_weather.solar_flux if latest_weather else None,
                } if latest_weather else None
            }

        except Exception as e:
            logger.error(f"Error getting collection health: {e}")
            return {
                "timestamp": datetime.utcnow().isoformat(),
                "health_score": 0,
                "error": str(e)
            }

    def close(self):
        """Close database connection if owned."""
        if self.owns_session and self.db:
            self.db.close()


# Utility functions
def get_recent_sessions(hours: int = 24) -> List[RecordingSession]:
    """Get recent recording sessions.

    Args:
        hours: Hours back to search

    Returns:
        List of recent sessions
    """
    db = MetadataDB()
    try:
        start_time = datetime.utcnow() - timedelta(hours=hours)
        return db.get_recording_sessions(start_time=start_time, limit=1000)
    finally:
        db.close()


def get_collection_summary() -> Dict[str, Any]:
    """Get overall collection summary.

    Returns:
        Collection summary
    """
    db = MetadataDB()
    try:
        return db.get_collection_statistics(days_back=30)
    finally:
        db.close()