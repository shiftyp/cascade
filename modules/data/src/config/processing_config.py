"""Post-processing pipeline configuration and versioning.

This file tracks the processing pipeline version and configuration.
When the pipeline changes significantly, increment PROCESSING_VERSION
to trigger reprocessing of existing recordings.
"""

# Processing pipeline version
# Increment this when making significant changes to processing algorithms
# that would benefit from reprocessing existing data
PROCESSING_VERSION = 1

# Version history:
# v1 (2025-01): Initial pipeline
#   - FT8 extraction with js8call decoder
#   - WSPR extraction with wsprd decoder
#   - Basic QRN analysis (SNR, power spectral density)
#   - Callsign anonymization via SHA-256
#
# v2 (Future): Enhanced pipeline
#   - Add JT65/JT9 extraction
#   - Improved QRN classification
#   - Multi-channel extraction for dense bands
#
# v3 (Future): ML-enhanced pipeline
#   - Neural network-based signal detection
#   - Automatic mode classification
#   - Anomaly detection

# Processing configuration
PROCESSING_CONFIG = {
    "version": PROCESSING_VERSION,

    # FT8 extraction settings
    "ft8": {
        "enabled": True,
        "decoder": "js8call",  # or "ft8_lib"
        "threads": 4,
        "min_snr": -24,  # Minimum SNR to report
    },

    # WSPR extraction settings
    "wspr": {
        "enabled": True,
        "decoder": "wsprd",
        "threads": 2,
        "min_snr": -30,
    },

    # QRN analysis settings
    "qrn": {
        "enabled": True,
        "window_seconds": 10,  # Analysis window size
        "overlap": 0.5,  # Window overlap ratio
        "metrics": [
            "snr",
            "power_spectral_density",
            "impulse_ratio",
            "spectral_occupancy",
        ],
    },

    # Storage settings
    "storage": {
        "keep_processed_locally": False,  # Delete local files after upload
        "compress_processed": True,  # FLAC compression
        "upload_to_tigris": True,
    },

    # Batch processing
    "batch": {
        "size": 100,  # Process recordings in batches
        "parallel_workers": 4,  # Parallel processing workers
        "retry_failed": True,
        "max_retries": 3,
    },
}

def should_reprocess(session_version: int) -> bool:
    """Check if a session needs reprocessing based on version.

    Args:
        session_version: Processing version applied to session

    Returns:
        True if session should be reprocessed
    """
    if session_version is None:
        return True  # Never processed
    return session_version < PROCESSING_VERSION

def get_unprocessed_query_filter():
    """Get SQLAlchemy filter for unprocessed sessions.

    Returns filter for sessions that need processing based on:
    - Never processed (processing_version is NULL)
    - Processed with older version
    - Failed processing that should be retried
    """
    from sqlalchemy import or_
    from ..models import RecordingSession

    return or_(
        RecordingSession.processing_version.is_(None),
        RecordingSession.processing_version < PROCESSING_VERSION,
        RecordingSession.processing_status == "failed",
    )