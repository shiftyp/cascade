#!/usr/bin/env python3
"""
Create missing database tables for CASCADE Data Collector.

This script creates tables that might be missing from the database.
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, '/app/src')

from sqlalchemy import text
from models.base import engine, SessionLocal
from models.collection_schedule import CollectionSchedule
from models.kiwisdr_source import KiwiSDRSource
from models.websdr_source import WebSDRSource
from models.recording_session import RecordingSession
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_missing_tables():
    """Create any missing database tables."""
    try:
        logger.info("🗄️  Creating missing database tables...")
        
        # Import all models to ensure they're registered
        from models import (
            Base, KiwiSDRSource, WebSDRSource, RecordingSession,
            CollectionSchedule, AtmosphericEvent, SpaceWeatherData,
            PropagationRecord, QRNSample, FT8Signal, WSPRSignal,
            NotificationConfig, NotificationTemplates, CollectionAlerts
        )
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ All tables created successfully")
        
        # Verify critical tables exist
        db = SessionLocal()
        try:
            # Test basic queries
            db.execute(text("SELECT COUNT(*) FROM kiwisdr_sources")).scalar()
            logger.info("✅ kiwisdr_sources table accessible")
            
            db.execute(text("SELECT COUNT(*) FROM collection_schedules")).scalar()
            logger.info("✅ collection_schedules table accessible")
            
            db.execute(text("SELECT COUNT(*) FROM recording_sessions")).scalar()
            logger.info("✅ recording_sessions table accessible")
            
        except Exception as e:
            logger.error(f"❌ Error testing tables: {e}")
            db.rollback()
            raise
        finally:
            db.close()
            
        logger.info("🎉 Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"❌ Error creating tables: {e}")
        sys.exit(1)

if __name__ == "__main__":
    create_missing_tables()