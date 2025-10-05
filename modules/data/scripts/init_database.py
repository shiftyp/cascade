#!/usr/bin/env python3
"""
Initialize CASCADE database with all tables from SQLAlchemy models.

This script creates all tables defined in the models without running migrations.
Use this for initial setup or development environments.
"""

import os
import sys
import logging
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import (
    Base,
    engine,
    SessionLocal,
    # Import all models to register them with Base
    RecordingSession,
    KiwiSDRSource,
    WebSDRSource,
    QRNSample,
    PropagationRecord,
    SpaceWeatherData,
    FT8Signal,
    WSPRSignal,
    AtmosphericEvent,
    CollectionSchedule,
    NotificationConfig,
    CollectionAlert,
    NotificationTemplate,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def init_database(drop_existing=False):
    """Initialize database with all tables.

    Args:
        drop_existing: If True, drop all existing tables first (DANGEROUS!)
    """
    try:
        if drop_existing:
            logger.warning("Dropping all existing tables...")
            Base.metadata.drop_all(bind=engine)
            logger.info("All tables dropped")

        logger.info("Creating all tables from models...")
        Base.metadata.create_all(bind=engine)
        logger.info("All tables created successfully")

        # Verify tables were created
        db = SessionLocal()
        try:
            # Test each model with a simple query
            models_to_test = [
                (RecordingSession, "recording_sessions"),
                (KiwiSDRSource, "kiwisdr_sources"),
                (WebSDRSource, "websdr_sources"),
                (QRNSample, "qrn_samples"),
                (PropagationRecord, "propagation_records"),
                (SpaceWeatherData, "space_weather_data"),
                (FT8Signal, "ft8_signals"),
                (WSPRSignal, "wspr_signals"),
                (AtmosphericEvent, "atmospheric_events"),
                (CollectionSchedule, "collection_schedules"),
                (NotificationConfig, "notification_configs"),
                (CollectionAlert, "collection_alerts"),
                (NotificationTemplate, "notification_templates"),
            ]

            logger.info("\nVerifying tables:")
            for model, table_name in models_to_test:
                count = db.query(model).count()
                logger.info(f"  ✓ {table_name}: {count} records")

        finally:
            db.close()

        logger.info("\nDatabase initialization complete!")
        return True

    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        return False


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Initialize CASCADE database tables from SQLAlchemy models"
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop all existing tables before creating (DANGEROUS!)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check if tables exist, don't create them",
    )

    args = parser.parse_args()

    if args.check_only:
        db = SessionLocal()
        try:
            # Check if tables exist
            from sqlalchemy import inspect
            inspector = inspect(engine)
            tables = inspector.get_table_names()

            expected_tables = [
                "recording_sessions",
                "kiwisdr_sources",
                "websdr_sources",
                "qrn_samples",
                "propagation_records",
                "space_weather_data",
                "ft8_signals",
                "wspr_signals",
                "atmospheric_events",
                "collection_schedules",
                "notification_configs",
                "collection_alerts",
                "notification_templates",
            ]

            logger.info("Checking tables:")
            for table in expected_tables:
                if table in tables:
                    logger.info(f"  ✓ {table} exists")
                else:
                    logger.warning(f"  ✗ {table} missing")

            missing = set(expected_tables) - set(tables)
            if missing:
                logger.error(f"\nMissing tables: {', '.join(missing)}")
                logger.info("Run without --check-only to create them")
                sys.exit(1)
            else:
                logger.info("\nAll tables exist!")

        finally:
            db.close()
    else:
        success = init_database(drop_existing=args.drop_existing)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()