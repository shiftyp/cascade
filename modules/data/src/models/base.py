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
    "postgresql://cascade:cascade@localhost:5432/cascade_data"
)

# Create engine
engine = create_engine(DATABASE_URL, echo=False)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()