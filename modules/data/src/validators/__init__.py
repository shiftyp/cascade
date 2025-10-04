"""Validators package for data quality and coverage validation.

This package provides comprehensive validation for:
- IQ recording quality (FR-013, FR-019)
- Geographic, temporal, and band coverage
- QA sampling for hot storage (FR-036)
- QA reporting and trend analysis (FR-037)
- Quarantine management for failed samples (FR-038)
"""

from .quality_check import QualityCheck, QualityMetrics, QualityStatus, check_recording_quality, quick_quality_check
from .coverage_check import (
    CoverageCheck,
    BandCoverage,
    GeographicCoverage,
    TemporalCoverage,
    CoverageReport,
    check_overall_coverage,
    check_band_progress,
)
from .qa_sampler import QASampler, QASample, run_qa_sampling, rotate_qa_samples
from .qa_reporter import QAReporter, QAReport, QAMetricsSummary, QATrendAnalysis, generate_daily_qa_report, generate_weekly_qa_report
from .quarantine_manager import (
    QuarantineManager,
    QuarantineRecord,
    QuarantineReason,
    QuarantineStatus,
    quarantine_low_quality_session,
    run_quarantine_maintenance,
)

__all__ = [
    # Quality check
    "QualityCheck",
    "QualityMetrics",
    "QualityStatus",
    "check_recording_quality",
    "quick_quality_check",
    # Coverage check
    "CoverageCheck",
    "BandCoverage",
    "GeographicCoverage",
    "TemporalCoverage",
    "CoverageReport",
    "check_overall_coverage",
    "check_band_progress",
    # QA sampler
    "QASampler",
    "QASample",
    "run_qa_sampling",
    "rotate_qa_samples",
    # QA reporter
    "QAReporter",
    "QAReport",
    "QAMetricsSummary",
    "QATrendAnalysis",
    "generate_daily_qa_report",
    "generate_weekly_qa_report",
    # Quarantine manager
    "QuarantineManager",
    "QuarantineRecord",
    "QuarantineReason",
    "QuarantineStatus",
    "quarantine_low_quality_session",
    "run_quarantine_maintenance",
]