"""QA sampling for hot storage retention.

Implements T038a: QA sampler (FR-036).
"""

import logging
import random
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QASample:
    """QA sample record."""

    session_id: str
    band: str
    frequency_khz: int
    timestamp: datetime
    file_path: str
    file_size_bytes: int
    quality_score: float
    sampling_reason: str
    priority: int
    expires_at: datetime


class QASampler:
    """QA sampler for 1% hot storage retention."""

    # Sampling configuration
    SAMPLING_RATE = 0.01  # 1% of recordings
    MAX_HOT_STORAGE_TB = 0.5  # 500 GB for QA samples
    SAMPLE_RETENTION_DAYS = 90  # Keep samples for 90 days
    MIN_SAMPLES_PER_BAND = 100  # Minimum samples per band

    def __init__(self, db_session=None):
        """Initialize QA sampler.

        Args:
            db_session: Optional database session
        """
        from ..models import SessionLocal

        self.db = db_session or SessionLocal()
        self.owns_session = db_session is None

        # Random seed for reproducibility
        random.seed(42)

    def select_samples(self, lookback_hours: int = 24) -> List[QASample]:
        """Select QA samples from recent recordings.

        Args:
            lookback_hours: Hours to look back for sampling

        Returns:
            List of selected QA samples
        """
        from ..models import RecordingSession

        try:
            # Get recent completed sessions
            cutoff_time = datetime.utcnow() - timedelta(hours=lookback_hours)

            sessions = (
                self.db.query(RecordingSession)
                .filter(
                    RecordingSession.start_time >= cutoff_time,
                    RecordingSession.processing_status == "completed",
                    RecordingSession.quality_score.isnot(None),
                )
                .all()
            )

            if not sessions:
                logger.info("No sessions available for QA sampling")
                return []

            # Stratified sampling by band/time/location
            samples = self._stratified_sample(sessions)

            # Add high-value samples (rare events)
            rare_samples = self._select_rare_events(sessions)
            samples.extend(rare_samples)

            # Remove duplicates
            samples = self._deduplicate_samples(samples)

            # Check storage limits
            samples = self._enforce_storage_limits(samples)

            logger.info(f"Selected {len(samples)} QA samples from {len(sessions)} sessions")
            return samples

        except Exception as e:
            logger.error(f"QA sampling failed: {e}")
            return []

    def _stratified_sample(self, sessions: List) -> List[QASample]:
        """Perform stratified sampling across bands, times, and locations.

        Args:
            sessions: List of RecordingSession objects

        Returns:
            List of QA samples
        """
        from collections import defaultdict

        # Group sessions by stratification keys
        strata = defaultdict(list)

        for session in sessions:
            # Create stratification key: band + hour + grid_prefix
            hour = session.start_time.hour
            grid_prefix = session.kiwisdr.grid_square[:2] if session.kiwisdr and session.kiwisdr.grid_square else "XX"

            key = f"{session.band}_{hour:02d}_{grid_prefix}"
            strata[key].append(session)

        # Sample from each stratum
        samples = []

        for stratum_key, stratum_sessions in strata.items():
            # Calculate sample size for this stratum
            sample_size = max(1, int(len(stratum_sessions) * self.SAMPLING_RATE))

            # Random sample
            sampled_sessions = random.sample(stratum_sessions, min(sample_size, len(stratum_sessions)))

            for session in sampled_sessions:
                samples.append(
                    QASample(
                        session_id=str(session.session_id),
                        band=session.band,
                        frequency_khz=session.center_frequency_hz // 1000,
                        timestamp=session.start_time,
                        file_path=session.iq_file_path,
                        file_size_bytes=session.file_size_bytes,
                        quality_score=session.quality_score,
                        sampling_reason="stratified_random",
                        priority=5,
                        expires_at=datetime.utcnow() + timedelta(days=self.SAMPLE_RETENTION_DAYS),
                    )
                )

        return samples

    def _select_rare_events(self, sessions: List) -> List[QASample]:
        """Select samples from rare propagation events.

        Args:
            sessions: List of RecordingSession objects

        Returns:
            List of high-value QA samples
        """
        rare_samples = []

        for session in sessions:
            # Check for rare event markers
            is_rare = False
            reason = ""

            # High K-index (geomagnetic storm)
            if session.k_index and session.k_index >= 7:
                is_rare = True
                reason = f"geomagnetic_storm_k{session.k_index}"

            # High solar flux
            elif session.solar_flux_index and session.solar_flux_index >= 200:
                is_rare = True
                reason = f"high_solar_flux_{session.solar_flux_index}"

            # Exceptional quality
            elif session.quality_score and session.quality_score >= 0.95:
                is_rare = True
                reason = "exceptional_quality"

            # High signal count (rare propagation)
            elif session.signal_count and session.signal_count >= 100:
                is_rare = True
                reason = f"high_activity_{session.signal_count}_signals"

            if is_rare:
                rare_samples.append(
                    QASample(
                        session_id=str(session.session_id),
                        band=session.band,
                        frequency_khz=session.center_frequency_hz // 1000,
                        timestamp=session.start_time,
                        file_path=session.iq_file_path,
                        file_size_bytes=session.file_size_bytes,
                        quality_score=session.quality_score,
                        sampling_reason=reason,
                        priority=10,  # High priority
                        expires_at=datetime.utcnow()
                        + timedelta(days=self.SAMPLE_RETENTION_DAYS * 2),  # Keep longer
                    )
                )

        return rare_samples

    def _deduplicate_samples(self, samples: List[QASample]) -> List[QASample]:
        """Remove duplicate samples.

        Args:
            samples: List of QA samples

        Returns:
            Deduplicated list
        """
        seen = set()
        unique_samples = []

        for sample in samples:
            if sample.session_id not in seen:
                seen.add(sample.session_id)
                unique_samples.append(sample)

        return unique_samples

    def _enforce_storage_limits(self, samples: List[QASample]) -> List[QASample]:
        """Enforce storage limits by prioritizing samples.

        Args:
            samples: List of QA samples

        Returns:
            Filtered list within storage limits
        """
        # Calculate total size
        total_size = sum(s.file_size_bytes for s in samples)
        max_size = int(self.MAX_HOT_STORAGE_TB * 1024**4)

        if total_size <= max_size:
            return samples

        # Sort by priority (high to low) then quality score
        samples.sort(key=lambda s: (-s.priority, -s.quality_score))

        # Keep samples until storage limit
        selected = []
        current_size = 0

        for sample in samples:
            if current_size + sample.file_size_bytes <= max_size:
                selected.append(sample)
                current_size += sample.file_size_bytes
            else:
                break

        logger.warning(f"Storage limit enforced: kept {len(selected)}/{len(samples)} samples")
        return selected

    def store_qa_samples(self, samples: List[QASample]) -> int:
        """Store QA sample metadata in database.

        Args:
            samples: List of QA samples

        Returns:
            Number of samples stored
        """
        from ..models import QASampleRecord

        try:
            stored_count = 0

            for sample in samples:
                # Check if already exists
                existing = (
                    self.db.query(QASampleRecord)
                    .filter(QASampleRecord.session_id == sample.session_id)
                    .first()
                )

                if not existing:
                    qa_record = QASampleRecord(
                        session_id=sample.session_id,
                        band=sample.band,
                        frequency_khz=sample.frequency_khz,
                        timestamp=sample.timestamp,
                        file_path=sample.file_path,
                        file_size_bytes=sample.file_size_bytes,
                        quality_score=sample.quality_score,
                        sampling_reason=sample.sampling_reason,
                        priority=sample.priority,
                        expires_at=sample.expires_at,
                    )

                    self.db.add(qa_record)
                    stored_count += 1

            self.db.commit()
            logger.info(f"Stored {stored_count} new QA samples")
            return stored_count

        except Exception as e:
            logger.error(f"Failed to store QA samples: {e}")
            self.db.rollback()
            return 0

    def rotate_samples(self) -> int:
        """Rotate expired QA samples.

        Returns:
            Number of samples removed
        """
        from ..models import QASampleRecord

        try:
            now = datetime.utcnow()

            # Find expired samples
            expired = self.db.query(QASampleRecord).filter(QASampleRecord.expires_at < now).all()

            removed_count = len(expired)

            # Delete expired samples
            for sample in expired:
                self.db.delete(sample)

            self.db.commit()

            logger.info(f"Rotated {removed_count} expired QA samples")
            return removed_count

        except Exception as e:
            logger.error(f"Failed to rotate QA samples: {e}")
            self.db.rollback()
            return 0

    def get_qa_samples(
        self,
        band: Optional[str] = None,
        min_quality: float = 0.0,
        limit: int = 100,
    ) -> List[QASample]:
        """Get QA samples with filters.

        Args:
            band: Optional band filter
            min_quality: Minimum quality score
            limit: Maximum results

        Returns:
            List of QA samples
        """
        from ..models import QASampleRecord

        try:
            query = self.db.query(QASampleRecord)

            if band:
                query = query.filter(QASampleRecord.band == band)

            if min_quality > 0:
                query = query.filter(QASampleRecord.quality_score >= min_quality)

            # Order by priority and timestamp
            samples = query.order_by(QASampleRecord.priority.desc(), QASampleRecord.timestamp.desc()).limit(limit).all()

            return [
                QASample(
                    session_id=str(s.session_id),
                    band=s.band,
                    frequency_khz=s.frequency_khz,
                    timestamp=s.timestamp,
                    file_path=s.file_path,
                    file_size_bytes=s.file_size_bytes,
                    quality_score=s.quality_score,
                    sampling_reason=s.sampling_reason,
                    priority=s.priority,
                    expires_at=s.expires_at,
                )
                for s in samples
            ]

        except Exception as e:
            logger.error(f"Failed to get QA samples: {e}")
            return []

    def get_storage_usage(self) -> Dict[str, Any]:
        """Get QA sample storage usage.

        Returns:
            Storage usage metrics
        """
        from ..models import QASampleRecord
        from sqlalchemy import func

        try:
            total_samples = self.db.query(func.count(QASampleRecord.sample_id)).scalar() or 0

            total_size = self.db.query(func.sum(QASampleRecord.file_size_bytes)).scalar() or 0

            max_size = int(self.MAX_HOT_STORAGE_TB * 1024**4)

            return {
                "total_samples": total_samples,
                "total_bytes": total_size,
                "total_gb": total_size / 1024**3,
                "limit_tb": self.MAX_HOT_STORAGE_TB,
                "usage_percent": (total_size / max_size) * 100 if max_size > 0 else 0,
            }

        except Exception as e:
            logger.error(f"Failed to get storage usage: {e}")
            return {"error": str(e)}

    def balance_band_samples(self) -> int:
        """Ensure minimum samples per band.

        Returns:
            Number of samples added
        """
        from ..models import RecordingSession, QASampleRecord
        from sqlalchemy import func

        try:
            # Get band counts
            band_counts = (
                self.db.query(QASampleRecord.band, func.count(QASampleRecord.sample_id))
                .group_by(QASampleRecord.band)
                .all()
            )

            bands_needing_samples = [band for band, count in band_counts if count < self.MIN_SAMPLES_PER_BAND]

            added_count = 0

            for band in bands_needing_samples:
                # Get recent sessions for this band
                sessions = (
                    self.db.query(RecordingSession)
                    .filter(
                        RecordingSession.band == band, RecordingSession.processing_status == "completed"
                    )
                    .order_by(RecordingSession.start_time.desc())
                    .limit(self.MIN_SAMPLES_PER_BAND)
                    .all()
                )

                # Sample to fill gap
                needed = self.MIN_SAMPLES_PER_BAND - len([b for b, c in band_counts if b == band][0])
                sampled = random.sample(sessions, min(needed, len(sessions)))

                # Store samples
                samples = [
                    QASample(
                        session_id=str(s.session_id),
                        band=s.band,
                        frequency_khz=s.center_frequency_hz // 1000,
                        timestamp=s.start_time,
                        file_path=s.iq_file_path,
                        file_size_bytes=s.file_size_bytes,
                        quality_score=s.quality_score,
                        sampling_reason="band_balance",
                        priority=7,
                        expires_at=datetime.utcnow() + timedelta(days=self.SAMPLE_RETENTION_DAYS),
                    )
                    for s in sampled
                ]

                added_count += self.store_qa_samples(samples)

            logger.info(f"Added {added_count} samples for band balancing")
            return added_count

        except Exception as e:
            logger.error(f"Failed to balance band samples: {e}")
            return 0

    def close(self):
        """Close database connection if owned."""
        if self.owns_session and self.db:
            self.db.close()


# Convenience functions
def run_qa_sampling(lookback_hours: int = 24) -> int:
    """Run QA sampling process.

    Args:
        lookback_hours: Hours to look back

    Returns:
        Number of samples created
    """
    sampler = QASampler()
    try:
        samples = sampler.select_samples(lookback_hours)
        return sampler.store_qa_samples(samples)
    finally:
        sampler.close()


def rotate_qa_samples() -> int:
    """Rotate expired QA samples.

    Returns:
        Number removed
    """
    sampler = QASampler()
    try:
        return sampler.rotate_samples()
    finally:
        sampler.close()