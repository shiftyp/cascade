"""Quarantine management for failed quality checks.

Implements T038c: Quarantine manager (FR-038).
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class QuarantineReason(Enum):
    """Reasons for quarantine."""

    LOW_QUALITY = "low_quality"
    IQ_IMBALANCE = "iq_imbalance"
    HIGH_CLIPPING = "high_clipping"
    DATA_CORRUPTION = "data_corruption"
    DROPOUTS = "dropouts"
    INVALID_METADATA = "invalid_metadata"
    SDR_MALFUNCTION = "sdr_malfunction"
    PROCESSING_ERROR = "processing_error"


class QuarantineStatus(Enum):
    """Quarantine status."""

    QUARANTINED = "quarantined"
    UNDER_REVIEW = "under_review"
    APPROVED = "approved"
    REJECTED = "rejected"
    RECOVERED = "recovered"


@dataclass
class QuarantineRecord:
    """Quarantine record."""

    quarantine_id: str
    session_id: str
    reason: str
    severity: str  # low, medium, high, critical
    quarantined_at: datetime
    file_path: str
    file_size_bytes: int
    quality_score: float
    failure_details: Dict[str, Any]
    status: str
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    resolution: Optional[str] = None
    notes: Optional[str] = None


class QuarantineManager:
    """Manager for failed quality check quarantine."""

    # Auto-recovery thresholds
    AUTO_RECOVERY_QUALITY = 0.7  # Quality score for auto-recovery
    QUARANTINE_RETENTION_DAYS = 180  # Keep quarantined samples for 180 days
    REVIEW_QUEUE_LIMIT = 100  # Max manual review queue size

    def __init__(self, db_session=None):
        """Initialize quarantine manager.

        Args:
            db_session: Optional database session
        """
        from ..models import SessionLocal

        self.db = db_session or SessionLocal()
        self.owns_session = db_session is None

    def quarantine_session(
        self,
        session_id: str,
        reason: QuarantineReason,
        severity: str,
        failure_details: Dict[str, Any],
    ) -> QuarantineRecord:
        """Quarantine a recording session.

        Args:
            session_id: Recording session ID
            reason: Quarantine reason
            severity: Severity level
            failure_details: Detailed failure information

        Returns:
            QuarantineRecord
        """
        from ..models import RecordingSession, QuarantinedSession
        from uuid import uuid4

        try:
            # Get recording session
            session = self.db.query(RecordingSession).filter(RecordingSession.session_id == session_id).first()

            if not session:
                raise ValueError(f"Session not found: {session_id}")

            # Create quarantine record
            quarantine_id = str(uuid4())

            quarantined = QuarantinedSession(
                quarantine_id=quarantine_id,
                session_id=session_id,
                reason=reason.value,
                severity=severity,
                quarantined_at=datetime.utcnow(),
                file_path=session.iq_file_path,
                file_size_bytes=session.file_size_bytes,
                quality_score=session.quality_score or 0.0,
                failure_details=failure_details,
                status=QuarantineStatus.QUARANTINED.value,
            )

            self.db.add(quarantined)

            # Update session status
            session.processing_status = "quarantined"

            self.db.commit()
            self.db.refresh(quarantined)

            logger.warning(
                f"Quarantined session {session_id}: {reason.value} (severity: {severity})"
            )

            return QuarantineRecord(
                quarantine_id=quarantine_id,
                session_id=session_id,
                reason=reason.value,
                severity=severity,
                quarantined_at=quarantined.quarantined_at,
                file_path=quarantined.file_path,
                file_size_bytes=quarantined.file_size_bytes,
                quality_score=quarantined.quality_score,
                failure_details=failure_details,
                status=quarantined.status,
            )

        except Exception as e:
            logger.error(f"Failed to quarantine session {session_id}: {e}")
            self.db.rollback()
            raise

    def get_quarantined_sessions(
        self,
        reason: Optional[QuarantineReason] = None,
        severity: Optional[str] = None,
        status: Optional[QuarantineStatus] = None,
        limit: int = 100,
    ) -> List[QuarantineRecord]:
        """Get quarantined sessions with filters.

        Args:
            reason: Optional reason filter
            severity: Optional severity filter
            status: Optional status filter
            limit: Maximum results

        Returns:
            List of quarantine records
        """
        from ..models import QuarantinedSession

        try:
            query = self.db.query(QuarantinedSession)

            if reason:
                query = query.filter(QuarantinedSession.reason == reason.value)

            if severity:
                query = query.filter(QuarantinedSession.severity == severity)

            if status:
                query = query.filter(QuarantinedSession.status == status.value)

            records = query.order_by(QuarantinedSession.quarantined_at.desc()).limit(limit).all()

            return [
                QuarantineRecord(
                    quarantine_id=str(r.quarantine_id),
                    session_id=str(r.session_id),
                    reason=r.reason,
                    severity=r.severity,
                    quarantined_at=r.quarantined_at,
                    file_path=r.file_path,
                    file_size_bytes=r.file_size_bytes,
                    quality_score=r.quality_score,
                    failure_details=r.failure_details or {},
                    status=r.status,
                    reviewed_by=r.reviewed_by,
                    reviewed_at=r.reviewed_at,
                    resolution=r.resolution,
                    notes=r.notes,
                )
                for r in records
            ]

        except Exception as e:
            logger.error(f"Failed to get quarantined sessions: {e}")
            return []

    def get_review_queue(self) -> List[QuarantineRecord]:
        """Get manual review queue.

        Returns:
            List of sessions needing review
        """
        return self.get_quarantined_sessions(
            status=QuarantineStatus.UNDER_REVIEW, limit=self.REVIEW_QUEUE_LIMIT
        )

    def move_to_review(self, quarantine_id: str) -> bool:
        """Move quarantined session to review queue.

        Args:
            quarantine_id: Quarantine ID

        Returns:
            True if successful
        """
        from ..models import QuarantinedSession

        try:
            record = (
                self.db.query(QuarantinedSession)
                .filter(QuarantinedSession.quarantine_id == quarantine_id)
                .first()
            )

            if not record:
                logger.warning(f"Quarantine record not found: {quarantine_id}")
                return False

            record.status = QuarantineStatus.UNDER_REVIEW.value
            self.db.commit()

            logger.info(f"Moved {quarantine_id} to review queue")
            return True

        except Exception as e:
            logger.error(f"Failed to move to review: {e}")
            self.db.rollback()
            return False

    def approve_quarantine(
        self, quarantine_id: str, reviewer: str, resolution: str, notes: Optional[str] = None
    ) -> bool:
        """Approve quarantine (sample remains quarantined).

        Args:
            quarantine_id: Quarantine ID
            reviewer: Reviewer name
            resolution: Resolution description
            notes: Optional notes

        Returns:
            True if successful
        """
        from ..models import QuarantinedSession

        try:
            record = (
                self.db.query(QuarantinedSession)
                .filter(QuarantinedSession.quarantine_id == quarantine_id)
                .first()
            )

            if not record:
                return False

            record.status = QuarantineStatus.APPROVED.value
            record.reviewed_by = reviewer
            record.reviewed_at = datetime.utcnow()
            record.resolution = resolution
            record.notes = notes

            self.db.commit()

            logger.info(f"Approved quarantine {quarantine_id} by {reviewer}")
            return True

        except Exception as e:
            logger.error(f"Failed to approve quarantine: {e}")
            self.db.rollback()
            return False

    def reject_quarantine(
        self, quarantine_id: str, reviewer: str, resolution: str, notes: Optional[str] = None
    ) -> bool:
        """Reject quarantine (sample was incorrectly quarantined).

        Args:
            quarantine_id: Quarantine ID
            reviewer: Reviewer name
            resolution: Resolution description
            notes: Optional notes

        Returns:
            True if successful
        """
        from ..models import QuarantinedSession, RecordingSession

        try:
            record = (
                self.db.query(QuarantinedSession)
                .filter(QuarantinedSession.quarantine_id == quarantine_id)
                .first()
            )

            if not record:
                return False

            # Update quarantine record
            record.status = QuarantineStatus.REJECTED.value
            record.reviewed_by = reviewer
            record.reviewed_at = datetime.utcnow()
            record.resolution = resolution
            record.notes = notes

            # Restore session status
            session = self.db.query(RecordingSession).filter(RecordingSession.session_id == record.session_id).first()

            if session:
                session.processing_status = "completed"

            self.db.commit()

            logger.info(f"Rejected quarantine {quarantine_id} by {reviewer}")
            return True

        except Exception as e:
            logger.error(f"Failed to reject quarantine: {e}")
            self.db.rollback()
            return False

    def attempt_recovery(self, quarantine_id: str) -> bool:
        """Attempt to recover quarantined sample.

        Args:
            quarantine_id: Quarantine ID

        Returns:
            True if recovered
        """
        from ..models import QuarantinedSession, RecordingSession

        try:
            record = (
                self.db.query(QuarantinedSession)
                .filter(QuarantinedSession.quarantine_id == quarantine_id)
                .first()
            )

            if not record:
                return False

            # Check if sample can be recovered
            can_recover = self._can_recover(record)

            if can_recover:
                record.status = QuarantineStatus.RECOVERED.value
                record.resolution = "auto_recovered"

                # Restore session
                session = self.db.query(RecordingSession).filter(RecordingSession.session_id == record.session_id).first()

                if session:
                    session.processing_status = "completed"

                self.db.commit()

                logger.info(f"Recovered quarantined session {quarantine_id}")
                return True
            else:
                logger.debug(f"Cannot recover {quarantine_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to attempt recovery: {e}")
            self.db.rollback()
            return False

    def _can_recover(self, record) -> bool:
        """Check if quarantined record can be recovered.

        Args:
            record: QuarantinedSession record

        Returns:
            True if can be recovered
        """
        # Check quality score
        if record.quality_score >= self.AUTO_RECOVERY_QUALITY:
            return True

        # Check if reason is minor
        minor_reasons = [
            QuarantineReason.LOW_QUALITY.value,
            QuarantineReason.DROPOUTS.value,
        ]

        if record.reason in minor_reasons and record.severity == "low":
            return True

        return False

    def auto_recovery_scan(self) -> int:
        """Scan quarantined samples for auto-recovery.

        Returns:
            Number of samples recovered
        """
        try:
            quarantined = self.get_quarantined_sessions(status=QuarantineStatus.QUARANTINED)

            recovered_count = 0

            for record in quarantined:
                if self.attempt_recovery(record.quarantine_id):
                    recovered_count += 1

            logger.info(f"Auto-recovery scan: recovered {recovered_count} samples")
            return recovered_count

        except Exception as e:
            logger.error(f"Auto-recovery scan failed: {e}")
            return 0

    def cleanup_old_quarantine(self) -> int:
        """Clean up old quarantine records.

        Returns:
            Number of records deleted
        """
        from ..models import QuarantinedSession

        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.QUARANTINE_RETENTION_DAYS)

            # Delete old approved/rejected records
            old_records = (
                self.db.query(QuarantinedSession)
                .filter(
                    QuarantinedSession.quarantined_at < cutoff_date,
                    QuarantinedSession.status.in_(
                        [QuarantineStatus.APPROVED.value, QuarantineStatus.REJECTED.value]
                    ),
                )
                .all()
            )

            deleted_count = len(old_records)

            for record in old_records:
                self.db.delete(record)

            self.db.commit()

            logger.info(f"Cleaned up {deleted_count} old quarantine records")
            return deleted_count

        except Exception as e:
            logger.error(f"Failed to cleanup old quarantine: {e}")
            self.db.rollback()
            return 0

    def get_quarantine_metrics(self) -> Dict[str, Any]:
        """Get quarantine metrics.

        Returns:
            Metrics dict
        """
        from ..models import QuarantinedSession
        from sqlalchemy import func

        try:
            # Total quarantined
            total = self.db.query(func.count(QuarantinedSession.quarantine_id)).scalar() or 0

            # By status
            status_counts = (
                self.db.query(QuarantinedSession.status, func.count(QuarantinedSession.quarantine_id))
                .group_by(QuarantinedSession.status)
                .all()
            )

            # By reason
            reason_counts = (
                self.db.query(QuarantinedSession.reason, func.count(QuarantinedSession.quarantine_id))
                .group_by(QuarantinedSession.reason)
                .all()
            )

            # By severity
            severity_counts = (
                self.db.query(QuarantinedSession.severity, func.count(QuarantinedSession.quarantine_id))
                .group_by(QuarantinedSession.severity)
                .all()
            )

            # Review queue size
            review_queue_size = (
                self.db.query(func.count(QuarantinedSession.quarantine_id))
                .filter(QuarantinedSession.status == QuarantineStatus.UNDER_REVIEW.value)
                .scalar()
                or 0
            )

            return {
                "total_quarantined": total,
                "status_distribution": {status: count for status, count in status_counts},
                "reason_distribution": {reason: count for reason, count in reason_counts},
                "severity_distribution": {severity: count for severity, count in severity_counts},
                "review_queue_size": review_queue_size,
            }

        except Exception as e:
            logger.error(f"Failed to get quarantine metrics: {e}")
            return {"error": str(e)}

    def export_quarantine_report(self, output_path: Path) -> bool:
        """Export quarantine report to CSV.

        Args:
            output_path: Output file path

        Returns:
            True if successful
        """
        import csv

        try:
            quarantined = self.get_quarantined_sessions(limit=10000)

            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", newline="") as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(
                    [
                        "Quarantine ID",
                        "Session ID",
                        "Reason",
                        "Severity",
                        "Quarantined At",
                        "Quality Score",
                        "Status",
                        "Reviewed By",
                        "Reviewed At",
                        "Resolution",
                    ]
                )

                # Data
                for record in quarantined:
                    writer.writerow(
                        [
                            record.quarantine_id,
                            record.session_id,
                            record.reason,
                            record.severity,
                            record.quarantined_at.isoformat(),
                            f"{record.quality_score:.3f}",
                            record.status,
                            record.reviewed_by or "",
                            record.reviewed_at.isoformat() if record.reviewed_at else "",
                            record.resolution or "",
                        ]
                    )

            logger.info(f"Exported quarantine report to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export quarantine report: {e}")
            return False

    def close(self):
        """Close database connection if owned."""
        if self.owns_session and self.db:
            self.db.close()


# Convenience functions
def quarantine_low_quality_session(session_id: str, quality_score: float, failure_details: Dict[str, Any]) -> bool:
    """Quarantine session due to low quality.

    Args:
        session_id: Session ID
        quality_score: Quality score
        failure_details: Failure details

    Returns:
        True if successful
    """
    manager = QuarantineManager()
    try:
        severity = "critical" if quality_score < 0.3 else "high" if quality_score < 0.5 else "medium"

        manager.quarantine_session(
            session_id, QuarantineReason.LOW_QUALITY, severity, failure_details
        )
        return True
    except Exception as e:
        logger.error(f"Failed to quarantine session: {e}")
        return False
    finally:
        manager.close()


def run_quarantine_maintenance() -> Dict[str, int]:
    """Run quarantine maintenance tasks.

    Returns:
        Dict of operation counts
    """
    manager = QuarantineManager()
    try:
        recovered = manager.auto_recovery_scan()
        cleaned = manager.cleanup_old_quarantine()

        return {"recovered": recovered, "cleaned": cleaned}
    finally:
        manager.close()