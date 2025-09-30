"""18-month collection window urgency monitoring.

Implements monitoring and alerting for the critical 18-month collection period.
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

from src.models import SessionLocal, SpaceWeatherData, RecordingSession
from src.analytics.rarity_scoring import RarityScorer

logger = logging.getLogger(__name__)


@dataclass
class UrgencyMetrics:
    """18-month collection window urgency metrics."""

    # Time tracking
    collection_window_start: datetime
    collection_window_end: datetime
    days_elapsed: int
    days_remaining: int
    completion_percentage: float

    # Collection metrics
    total_hours_collected: float
    target_hours: float
    collection_rate_per_day: float
    projected_total_hours: float

    # Urgency indicators
    urgency_level: str  # LOW, MODERATE, HIGH, CRITICAL
    time_pressure_factor: float
    behind_schedule: bool
    days_behind: int

    # Event metrics
    rare_events_captured: int
    missed_opportunities: int
    solar_minimum_events: int

    # Recommendations
    recommended_action: str
    scaling_recommendation: int
    priority_bands: List[str]


class UrgencyMonitor:
    """Monitors 18-month collection window urgency and provides alerts."""

    def __init__(self):
        """Initialize urgency monitor."""
        self.db = SessionLocal()
        self.rarity_scorer = RarityScorer()

        # Collection window parameters
        self.collection_start = datetime(2024, 12, 1)
        self.collection_end = datetime(2026, 6, 1)
        self.target_total_hours = 150000  # 150,000-300,000 hours target
        self.baseline_hours_per_day = 280  # ~280 hours/day target

    async def get_urgency_metrics(self) -> UrgencyMetrics:
        """Calculate current urgency metrics for 18-month window.

        Returns:
            Current urgency metrics
        """
        current_time = datetime.utcnow()

        # Time calculations
        total_days = (self.collection_end - self.collection_start).days
        days_elapsed = (current_time - self.collection_start).days
        days_remaining = (self.collection_end - current_time).days
        completion_percentage = (days_elapsed / total_days) * 100

        # Collection metrics
        collection_stats = await self._get_collection_statistics()
        total_hours = collection_stats["total_hours"]
        rate_per_day = collection_stats["daily_rate"]
        projected_total = total_hours + (rate_per_day * days_remaining)

        # Calculate urgency level
        urgency_data = self._calculate_urgency_level(
            completion_percentage, total_hours, projected_total, days_remaining
        )

        # Event tracking
        event_stats = await self._get_event_statistics()

        return UrgencyMetrics(
            collection_window_start=self.collection_start,
            collection_window_end=self.collection_end,
            days_elapsed=max(0, days_elapsed),
            days_remaining=max(0, days_remaining),
            completion_percentage=completion_percentage,
            total_hours_collected=total_hours,
            target_hours=self.target_total_hours,
            collection_rate_per_day=rate_per_day,
            projected_total_hours=projected_total,
            urgency_level=urgency_data["level"],
            time_pressure_factor=urgency_data["pressure_factor"],
            behind_schedule=urgency_data["behind_schedule"],
            days_behind=urgency_data["days_behind"],
            rare_events_captured=event_stats["rare_events"],
            missed_opportunities=event_stats["missed_events"],
            solar_minimum_events=event_stats["solar_min_events"],
            recommended_action=urgency_data["action"],
            scaling_recommendation=urgency_data["recommended_sdrs"],
            priority_bands=urgency_data["priority_bands"],
        )

    async def _get_collection_statistics(self) -> Dict[str, float]:
        """Get collection statistics.

        Returns:
            Collection statistics
        """
        try:
            # Get all recording sessions in the collection window
            sessions = (
                self.db.query(RecordingSession)
                .filter(
                    RecordingSession.start_time >= self.collection_start,
                    RecordingSession.end_time <= datetime.utcnow(),
                    RecordingSession.end_time.isnot(None),
                )
                .all()
            )

            # Calculate total hours
            total_hours = 0.0
            for session in sessions:
                if session.end_time and session.start_time:
                    duration = (session.end_time - session.start_time).total_seconds() / 3600
                    total_hours += duration

            # Calculate daily rate (last 7 days)
            week_ago = datetime.utcnow() - timedelta(days=7)
            recent_sessions = [
                s for s in sessions
                if s.start_time >= week_ago
            ]

            recent_hours = 0.0
            for session in recent_sessions:
                if session.end_time and session.start_time:
                    duration = (session.end_time - session.start_time).total_seconds() / 3600
                    recent_hours += duration

            daily_rate = recent_hours / 7.0  # Average over 7 days

            return {
                "total_hours": total_hours,
                "daily_rate": daily_rate,
                "session_count": len(sessions),
            }

        except Exception as e:
            logger.error(f"Error getting collection statistics: {e}")
            return {"total_hours": 0.0, "daily_rate": 0.0, "session_count": 0}

    def _calculate_urgency_level(
        self,
        completion_percentage: float,
        total_hours: float,
        projected_total: float,
        days_remaining: int,
    ) -> Dict[str, Any]:
        """Calculate urgency level and recommendations.

        Args:
            completion_percentage: Time completion percentage
            total_hours: Hours collected so far
            projected_total: Projected total hours
            days_remaining: Days remaining

        Returns:
            Urgency analysis
        """
        # Calculate schedule adherence
        expected_hours = (completion_percentage / 100) * self.target_total_hours
        hours_behind = expected_hours - total_hours
        days_behind = hours_behind / self.baseline_hours_per_day

        behind_schedule = hours_behind > (self.baseline_hours_per_day * 3)  # 3+ days behind

        # Time pressure factor
        if days_remaining < 30:
            pressure_factor = 10.0  # Critical final month
        elif days_remaining < 90:
            pressure_factor = 5.0   # High pressure final quarter
        elif days_remaining < 180:
            pressure_factor = 3.0   # Moderate pressure
        else:
            pressure_factor = 1.5   # Early phase

        # Additional pressure if behind schedule
        if behind_schedule:
            pressure_factor *= 2.0

        # Determine urgency level
        if pressure_factor >= 8.0 or days_remaining < 30:
            urgency_level = "CRITICAL"
            recommended_sdrs = 50
            action = "MAXIMUM DEPLOYMENT - Deploy all available SDRs immediately"
            priority_bands = ["20m", "40m", "80m", "15m", "10m", "6m"]
        elif pressure_factor >= 5.0 or behind_schedule:
            urgency_level = "HIGH"
            recommended_sdrs = 35
            action = "AGGRESSIVE SCALING - Increase collection intensity"
            priority_bands = ["20m", "40m", "80m", "15m"]
        elif pressure_factor >= 3.0:
            urgency_level = "MODERATE"
            recommended_sdrs = 20
            action = "ENHANCED COLLECTION - Scale up during events"
            priority_bands = ["20m", "40m", "80m"]
        else:
            urgency_level = "LOW"
            recommended_sdrs = 12
            action = "BASELINE COLLECTION - Maintain current schedule"
            priority_bands = ["20m", "40m"]

        return {
            "level": urgency_level,
            "pressure_factor": pressure_factor,
            "behind_schedule": behind_schedule,
            "days_behind": max(0, int(days_behind)),
            "action": action,
            "recommended_sdrs": recommended_sdrs,
            "priority_bands": priority_bands,
        }

    async def _get_event_statistics(self) -> Dict[str, int]:
        """Get rare event capture statistics.

        Returns:
            Event statistics
        """
        try:
            # Get space weather events in collection window
            events = (
                self.db.query(SpaceWeatherData)
                .filter(
                    SpaceWeatherData.observation_time >= self.collection_start,
                    SpaceWeatherData.observation_time <= datetime.utcnow(),
                )
                .all()
            )

            rare_events = 0
            solar_min_events = 0
            missed_events = 0

            for event in events:
                # Count rare events (high rarity multiplier)
                if event.get_rarity_multiplier() >= 5.0:
                    rare_events += 1

                # Count solar minimum events
                if event.is_solar_minimum_compensation_active():
                    if (event.k_index and event.k_index >= 3) or event.should_include_c_flares():
                        solar_min_events += 1

                # Count potentially missed events (100% capture required but low collection)
                if event.is_100_percent_capture_required():
                    # Check if we had enough collection sessions during this event
                    event_time = event.observation_time
                    event_sessions = (
                        self.db.query(RecordingSession)
                        .filter(
                            RecordingSession.start_time >= event_time - timedelta(hours=2),
                            RecordingSession.start_time <= event_time + timedelta(hours=2),
                        )
                        .count()
                    )

                    # If fewer than 20 sessions during 100% capture event, consider it missed
                    if event_sessions < 20:
                        missed_events += 1

            return {
                "rare_events": rare_events,
                "missed_events": missed_events,
                "solar_min_events": solar_min_events,
            }

        except Exception as e:
            logger.error(f"Error getting event statistics: {e}")
            return {"rare_events": 0, "missed_events": 0, "solar_min_events": 0}

    async def check_critical_alerts(self) -> List[Dict[str, Any]]:
        """Check for critical urgency alerts.

        Returns:
            List of active alerts
        """
        metrics = await self.get_urgency_metrics()
        alerts = []

        # Time-based alerts
        if metrics.days_remaining < 30:
            alerts.append({
                "type": "CRITICAL_TIME",
                "severity": "CRITICAL",
                "message": f"Only {metrics.days_remaining} days remaining in collection window",
                "action": "Deploy maximum resources immediately",
            })

        elif metrics.days_remaining < 90:
            alerts.append({
                "type": "TIME_WARNING",
                "severity": "HIGH",
                "message": f"{metrics.days_remaining} days remaining - entering final quarter",
                "action": "Increase collection intensity",
            })

        # Schedule-based alerts
        if metrics.behind_schedule:
            alerts.append({
                "type": "BEHIND_SCHEDULE",
                "severity": "HIGH",
                "message": f"{metrics.days_behind} days behind collection schedule",
                "action": f"Scale to {metrics.scaling_recommendation} SDRs",
            })

        # Collection rate alerts
        target_rate = self.baseline_hours_per_day
        if metrics.collection_rate_per_day < target_rate * 0.5:
            alerts.append({
                "type": "LOW_COLLECTION_RATE",
                "severity": "CRITICAL",
                "message": f"Collection rate {metrics.collection_rate_per_day:.1f} h/day is critically low (target: {target_rate})",
                "action": "Investigate system issues and scale up immediately",
            })

        elif metrics.collection_rate_per_day < target_rate * 0.7:
            alerts.append({
                "type": "SUBOPTIMAL_RATE",
                "severity": "MODERATE",
                "message": f"Collection rate {metrics.collection_rate_per_day:.1f} h/day below target ({target_rate})",
                "action": "Consider scaling up collection",
            })

        # Missed opportunity alerts
        if metrics.missed_opportunities > 5:
            alerts.append({
                "type": "MISSED_OPPORTUNITIES",
                "severity": "HIGH",
                "message": f"{metrics.missed_opportunities} rare events potentially missed",
                "action": "Review 100% capture enforcement",
            })

        return alerts

    async def generate_daily_report(self) -> Dict[str, Any]:
        """Generate daily urgency report.

        Returns:
            Daily report
        """
        metrics = await self.get_urgency_metrics()
        alerts = await self.check_critical_alerts()

        # Get recent space weather
        latest_weather = (
            self.db.query(SpaceWeatherData)
            .order_by(SpaceWeatherData.observation_time.desc())
            .first()
        )

        current_priority = None
        if latest_weather:
            current_priority = await self.rarity_scorer.get_collection_priority(latest_weather)

        return {
            "report_date": datetime.utcnow().isoformat(),
            "collection_window_status": {
                "days_remaining": metrics.days_remaining,
                "completion_percentage": metrics.completion_percentage,
                "urgency_level": metrics.urgency_level,
                "time_pressure_factor": metrics.time_pressure_factor,
            },
            "collection_progress": {
                "total_hours": metrics.total_hours_collected,
                "target_hours": metrics.target_hours,
                "progress_percentage": (metrics.total_hours_collected / metrics.target_hours) * 100,
                "projected_total": metrics.projected_total_hours,
                "daily_rate": metrics.collection_rate_per_day,
                "behind_schedule": metrics.behind_schedule,
                "days_behind": metrics.days_behind,
            },
            "current_conditions": current_priority,
            "recommendations": {
                "action": metrics.recommended_action,
                "sdr_target": metrics.scaling_recommendation,
                "priority_bands": metrics.priority_bands,
            },
            "alerts": alerts,
            "event_tracking": {
                "rare_events_captured": metrics.rare_events_captured,
                "solar_minimum_events": metrics.solar_minimum_events,
                "missed_opportunities": metrics.missed_opportunities,
            },
        }

    def close(self):
        """Close monitor resources."""
        if self.rarity_scorer:
            self.rarity_scorer.close()
        if self.db:
            self.db.close()


async def get_urgency_dashboard() -> Dict[str, Any]:
    """Get urgency dashboard data (utility function).

    Returns:
        Dashboard data
    """
    monitor = UrgencyMonitor()

    try:
        return await monitor.generate_daily_report()
    finally:
        monitor.close()