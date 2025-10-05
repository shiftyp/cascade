"""Base configuration for SQLAlchemy models."""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

# Create base class for models - singleton pattern to avoid duplicate tables
_Base = None

def get_base():
    """Get or create the declarative base."""
    global _Base
    if _Base is None:
        _Base = declarative_base()
    return _Base

Base = get_base()

# Database configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@cascade-rf-db.internal:5432/cascade_data"
)

# Ensure proper SQLAlchemy 2.x compatibility
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg2://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg2" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg2://", 1)

# Create engine with larger connection pool for concurrent recordings
engine = create_engine(
    DATABASE_URL,
    echo=False,
    pool_size=20,  # Increased from default 5
    max_overflow=30,  # Increased from default 10
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()