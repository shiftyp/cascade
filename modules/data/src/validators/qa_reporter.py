"""QA report generator for quality metrics and trends.

Implements T038b: QA reporter (FR-037).
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class QAMetricsSummary:
    """Summary of QA metrics."""

    period: str
    start_date: datetime
    end_date: datetime
    total_samples: int
    avg_quality: float
    min_quality: float
    max_quality: float
    quality_distribution: Dict[str, int]
    band_breakdown: Dict[str, Dict[str, Any]]


@dataclass
class QATrendAnalysis:
    """QA trend analysis."""

    metric: str
    current_value: float
    previous_value: float
    change_percent: float
    trend: str  # improving, declining, stable


@dataclass
class QAReport:
    """Comprehensive QA report."""

    report_type: str  # daily, weekly, monthly
    generated_at: datetime
    period_start: datetime
    period_end: datetime
    metrics_summary: QAMetricsSummary
    trends: List[QATrendAnalysis]
    coverage_heatmap: Dict[str, Any]
    alerts: List[str]
    recommendations: List[str]


class QAReporter:
    """QA report generator for quality monitoring."""

    def __init__(self, db_session=None):
        """Initialize QA reporter.

        Args:
            db_session: Optional database session
        """
        from ..models import SessionLocal

        self.db = db_session or SessionLocal()
        self.owns_session = db_session is None

    def generate_daily_report(self, date: Optional[datetime] = None) -> QAReport:
        """Generate daily QA report.

        Args:
            date: Report date (defaults to yesterday)

        Returns:
            Daily QA report
        """
        if not date:
            date = datetime.utcnow() - timedelta(days=1)

        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        return self._generate_report("daily", start, end)

    def generate_weekly_report(self, date: Optional[datetime] = None) -> QAReport:
        """Generate weekly QA report.

        Args:
            date: Week ending date (defaults to last week)

        Returns:
            Weekly QA report
        """
        if not date:
            date = datetime.utcnow() - timedelta(days=7)

        end = date.replace(hour=0, minute=0, second=0, microsecond=0)
        start = end - timedelta(days=7)

        return self._generate_report("weekly", start, end)

    def generate_monthly_report(self, year: Optional[int] = None, month: Optional[int] = None) -> QAReport:
        """Generate monthly QA report.

        Args:
            year: Report year (defaults to last month)
            month: Report month (defaults to last month)

        Returns:
            Monthly QA report
        """
        if not year or not month:
            last_month = datetime.utcnow().replace(day=1) - timedelta(days=1)
            year = last_month.year
            month = last_month.month

        start = datetime(year, month, 1)
        if month == 12:
            end = datetime(year + 1, 1, 1)
        else:
            end = datetime(year, month + 1, 1)

        return self._generate_report("monthly", start, end)

    def _generate_report(self, report_type: str, start: datetime, end: datetime) -> QAReport:
        """Generate QA report for period.

        Args:
            report_type: Report type
            start: Period start
            end: Period end

        Returns:
            QA report
        """
        try:
            # Generate metrics summary
            metrics = self._calculate_metrics(start, end)

            # Analyze trends
            trends = self._analyze_trends(report_type, start, end)

            # Generate coverage heatmap
            heatmap = self._generate_coverage_heatmap(start, end)

            # Detect quality issues
            alerts = self._detect_quality_alerts(metrics, trends)

            # Generate recommendations
            recommendations = self._generate_recommendations(metrics, trends, alerts)

            return QAReport(
                report_type=report_type,
                generated_at=datetime.utcnow(),
                period_start=start,
                period_end=end,
                metrics_summary=metrics,
                trends=trends,
                coverage_heatmap=heatmap,
                alerts=alerts,
                recommendations=recommendations,
            )

        except Exception as e:
            logger.error(f"Failed to generate {report_type} report: {e}")
            raise

    def _calculate_metrics(self, start: datetime, end: datetime) -> QAMetricsSummary:
        """Calculate QA metrics for period.

        Args:
            start: Period start
            end: Period end

        Returns:
            Metrics summary
        """
        from ..models import RecordingSession
        from sqlalchemy import func

        try:
            # Get sessions in period
            sessions = (
                self.db.query(RecordingSession)
                .filter(
                    RecordingSession.start_time >= start,
                    RecordingSession.start_time < end,
                    RecordingSession.processing_status == "completed",
                    RecordingSession.quality_score.isnot(None),
                )
                .all()
            )

            if not sessions:
                return QAMetricsSummary(
                    period=f"{start.date()} to {end.date()}",
                    start_date=start,
                    end_date=end,
                    total_samples=0,
                    avg_quality=0.0,
                    min_quality=0.0,
                    max_quality=0.0,
                    quality_distribution={},
                    band_breakdown={},
                )

            # Quality statistics
            quality_scores = [s.quality_score for s in sessions]
            avg_quality = sum(quality_scores) / len(quality_scores)
            min_quality = min(quality_scores)
            max_quality = max(quality_scores)

            # Quality distribution (bins: poor, fair, good, excellent)
            distribution = {"poor": 0, "fair": 0, "good": 0, "excellent": 0}

            for score in quality_scores:
                if score < 0.5:
                    distribution["poor"] += 1
                elif score < 0.7:
                    distribution["fair"] += 1
                elif score < 0.9:
                    distribution["good"] += 1
                else:
                    distribution["excellent"] += 1

            # Band breakdown
            band_breakdown = self._calculate_band_breakdown(sessions)

            return QAMetricsSummary(
                period=f"{start.date()} to {end.date()}",
                start_date=start,
                end_date=end,
                total_samples=len(sessions),
                avg_quality=avg_quality,
                min_quality=min_quality,
                max_quality=max_quality,
                quality_distribution=distribution,
                band_breakdown=band_breakdown,
            )

        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            raise

    def _calculate_band_breakdown(self, sessions: List) -> Dict[str, Dict[str, Any]]:
        """Calculate per-band metrics.

        Args:
            sessions: List of RecordingSession objects

        Returns:
            Band breakdown dict
        """
        from collections import defaultdict

        band_data = defaultdict(lambda: {"count": 0, "quality_scores": []})

        for session in sessions:
            band_data[session.band]["count"] += 1
            band_data[session.band]["quality_scores"].append(session.quality_score)

        # Calculate averages
        breakdown = {}
        for band, data in band_data.items():
            scores = data["quality_scores"]
            breakdown[band] = {
                "sample_count": data["count"],
                "avg_quality": sum(scores) / len(scores),
                "min_quality": min(scores),
                "max_quality": max(scores),
            }

        return breakdown

    def _analyze_trends(self, report_type: str, start: datetime, end: datetime) -> List[QATrendAnalysis]:
        """Analyze quality trends.

        Args:
            report_type: Report type
            start: Current period start
            end: Current period end

        Returns:
            List of trend analyses
        """
        from ..models import RecordingSession
        from sqlalchemy import func

        try:
            # Calculate period length
            period_delta = end - start

            # Previous period
            prev_start = start - period_delta
            prev_end = start

            # Current period metrics
            current_avg = (
                self.db.query(func.avg(RecordingSession.quality_score))
                .filter(
                    RecordingSession.start_time >= start,
                    RecordingSession.start_time < end,
                    RecordingSession.quality_score.isnot(None),
                )
                .scalar()
                or 0.0
            )

            # Previous period metrics
            prev_avg = (
                self.db.query(func.avg(RecordingSession.quality_score))
                .filter(
                    RecordingSession.start_time >= prev_start,
                    RecordingSession.start_time < prev_end,
                    RecordingSession.quality_score.isnot(None),
                )
                .scalar()
                or 0.0
            )

            # Calculate change
            if prev_avg > 0:
                change_percent = ((current_avg - prev_avg) / prev_avg) * 100
            else:
                change_percent = 0.0

            # Determine trend
            if abs(change_percent) < 2:
                trend = "stable"
            elif change_percent > 0:
                trend = "improving"
            else:
                trend = "declining"

            trends = [
                QATrendAnalysis(
                    metric="average_quality",
                    current_value=float(current_avg),
                    previous_value=float(prev_avg),
                    change_percent=change_percent,
                    trend=trend,
                )
            ]

            return trends

        except Exception as e:
            logger.error(f"Failed to analyze trends: {e}")
            return []

    def _generate_coverage_heatmap(self, start: datetime, end: datetime) -> Dict[str, Any]:
        """Generate coverage heatmap data.

        Args:
            start: Period start
            end: Period end

        Returns:
            Heatmap data
        """
        from ..models import RecordingSession
        from sqlalchemy import func, extract

        try:
            # Hour x Band heatmap
            heatmap_data = (
                self.db.query(
                    RecordingSession.band,
                    extract("hour", RecordingSession.start_time).label("hour"),
                    func.count(RecordingSession.session_id).label("count"),
                    func.avg(RecordingSession.quality_score).label("avg_quality"),
                )
                .filter(
                    RecordingSession.start_time >= start,
                    RecordingSession.start_time < end,
                    RecordingSession.processing_status == "completed",
                )
                .group_by(RecordingSession.band, "hour")
                .all()
            )

            # Format as nested dict
            heatmap = {}
            for band, hour, count, avg_quality in heatmap_data:
                if band not in heatmap:
                    heatmap[band] = {}

                heatmap[band][int(hour)] = {"sample_count": count, "avg_quality": float(avg_quality or 0)}

            return heatmap

        except Exception as e:
            logger.error(f"Failed to generate coverage heatmap: {e}")
            return {}

    def _detect_quality_alerts(self, metrics: QAMetricsSummary, trends: List[QATrendAnalysis]) -> List[str]:
        """Detect quality degradation alerts.

        Args:
            metrics: Metrics summary
            trends: Trend analyses

        Returns:
            List of alert messages
        """
        alerts = []

        # Check average quality
        if metrics.avg_quality < 0.6:
            alerts.append(f"Low average quality: {metrics.avg_quality:.2f}")

        # Check quality distribution
        if metrics.quality_distribution.get("poor", 0) > metrics.total_samples * 0.1:
            poor_percent = (metrics.quality_distribution["poor"] / metrics.total_samples) * 100
            alerts.append(f"High poor quality rate: {poor_percent:.1f}%")

        # Check trends
        for trend in trends:
            if trend.trend == "declining" and abs(trend.change_percent) > 10:
                alerts.append(f"{trend.metric} declining: {trend.change_percent:.1f}% decrease")

        # Check band coverage
        for band, data in metrics.band_breakdown.items():
            if data["avg_quality"] < 0.5:
                alerts.append(f"Band {band} quality degraded: {data['avg_quality']:.2f}")

        return alerts

    def _generate_recommendations(
        self, metrics: QAMetricsSummary, trends: List[QATrendAnalysis], alerts: List[str]
    ) -> List[str]:
        """Generate recommendations based on QA analysis.

        Args:
            metrics: Metrics summary
            trends: Trend analyses
            alerts: Alert messages

        Returns:
            List of recommendations
        """
        recommendations = []

        # Low quality recommendations
        if metrics.avg_quality < 0.7:
            recommendations.append("Review SDR selection criteria - prioritize high-reliability receivers")
            recommendations.append("Increase minimum quality threshold for recording retention")

        # Declining trend recommendations
        for trend in trends:
            if trend.trend == "declining":
                recommendations.append(f"Investigate {trend.metric} decline - check SDR health monitoring")

        # Band-specific recommendations
        for band, data in metrics.band_breakdown.items():
            if data["sample_count"] < 100:
                recommendations.append(f"Increase collection priority for {band} - low sample count")

            if data["avg_quality"] < 0.6:
                recommendations.append(f"Review SDR configuration for {band} recordings")

        # If no issues, provide optimization tips
        if not alerts and not recommendations:
            recommendations.append("Quality metrics healthy - continue current collection strategy")

        return recommendations

    def export_report(self, report: QAReport, output_path: Path, format: str = "json") -> bool:
        """Export report to file.

        Args:
            report: QA report
            output_path: Output file path
            format: Export format (json, markdown)

        Returns:
            True if successful
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "json":
                self._export_json(report, output_path)
            elif format == "markdown":
                self._export_markdown(report, output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Exported {report.report_type} report to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export report: {e}")
            return False

    def _export_json(self, report: QAReport, output_path: Path):
        """Export report as JSON.

        Args:
            report: QA report
            output_path: Output path
        """
        report_dict = {
            "report_type": report.report_type,
            "generated_at": report.generated_at.isoformat(),
            "period_start": report.period_start.isoformat(),
            "period_end": report.period_end.isoformat(),
            "metrics_summary": asdict(report.metrics_summary),
            "trends": [asdict(t) for t in report.trends],
            "coverage_heatmap": report.coverage_heatmap,
            "alerts": report.alerts,
            "recommendations": report.recommendations,
        }

        # Convert datetime objects
        report_dict["metrics_summary"]["start_date"] = report_dict["metrics_summary"]["start_date"].isoformat()
        report_dict["metrics_summary"]["end_date"] = report_dict["metrics_summary"]["end_date"].isoformat()

        with open(output_path, "w") as f:
            json.dump(report_dict, f, indent=2)

    def _export_markdown(self, report: QAReport, output_path: Path):
        """Export report as Markdown.

        Args:
            report: QA report
            output_path: Output path
        """
        lines = [
            f"# {report.report_type.title()} QA Report",
            f"",
            f"**Generated:** {report.generated_at.strftime('%Y-%m-%d %H:%M UTC')}",
            f"**Period:** {report.period_start.date()} to {report.period_end.date()}",
            f"",
            f"## Metrics Summary",
            f"",
            f"- **Total Samples:** {report.metrics_summary.total_samples}",
            f"- **Average Quality:** {report.metrics_summary.avg_quality:.3f}",
            f"- **Quality Range:** {report.metrics_summary.min_quality:.3f} - {report.metrics_summary.max_quality:.3f}",
            f"",
            f"### Quality Distribution",
            f"",
        ]

        for category, count in report.metrics_summary.quality_distribution.items():
            percent = (count / report.metrics_summary.total_samples) * 100 if report.metrics_summary.total_samples > 0 else 0
            lines.append(f"- **{category.title()}:** {count} ({percent:.1f}%)")

        lines.extend([f"", f"### Band Breakdown", f""])

        for band, data in report.metrics_summary.band_breakdown.items():
            lines.extend(
                [
                    f"**{band}**",
                    f"- Samples: {data['sample_count']}",
                    f"- Avg Quality: {data['avg_quality']:.3f}",
                    f"- Range: {data['min_quality']:.3f} - {data['max_quality']:.3f}",
                    f"",
                ]
            )

        lines.extend([f"## Trends", f""])

        for trend in report.trends:
            lines.extend(
                [
                    f"**{trend.metric.replace('_', ' ').title()}**",
                    f"- Current: {trend.current_value:.3f}",
                    f"- Previous: {trend.previous_value:.3f}",
                    f"- Change: {trend.change_percent:+.1f}%",
                    f"- Trend: {trend.trend}",
                    f"",
                ]
            )

        if report.alerts:
            lines.extend([f"## Alerts", f""])
            for alert in report.alerts:
                lines.append(f"- ⚠️ {alert}")
            lines.append("")

        if report.recommendations:
            lines.extend([f"## Recommendations", f""])
            for rec in report.recommendations:
                lines.append(f"- {rec}")

        with open(output_path, "w") as f:
            f.write("\n".join(lines))

    def close(self):
        """Close database connection if owned."""
        if self.owns_session and self.db:
            self.db.close()


# Convenience functions
def generate_daily_qa_report(output_dir: Optional[Path] = None) -> QAReport:
    """Generate daily QA report.

    Args:
        output_dir: Optional output directory for export

    Returns:
        QA report
    """
    reporter = QAReporter()
    try:
        report = reporter.generate_daily_report()

        if output_dir:
            date_str = report.period_start.strftime("%Y-%m-%d")
            json_path = output_dir / f"qa_daily_{date_str}.json"
            md_path = output_dir / f"qa_daily_{date_str}.md"

            reporter.export_report(report, json_path, "json")
            reporter.export_report(report, md_path, "markdown")

        return report
    finally:
        reporter.close()


def generate_weekly_qa_report(output_dir: Optional[Path] = None) -> QAReport:
    """Generate weekly QA report.

    Args:
        output_dir: Optional output directory

    Returns:
        QA report
    """
    reporter = QAReporter()
    try:
        report = reporter.generate_weekly_report()

        if output_dir:
            date_str = report.period_end.strftime("%Y-W%W")
            json_path = output_dir / f"qa_weekly_{date_str}.json"
            md_path = output_dir / f"qa_weekly_{date_str}.md"

            reporter.export_report(report, json_path, "json")
            reporter.export_report(report, md_path, "markdown")

        return report
    finally:
        reporter.close()