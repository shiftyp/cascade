"""Main FastAPI application for CASCADE Data Collector."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import collectors, processors
from ..config import config
from ..models import engine, Base

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting CASCADE Data Collector API")

    # Create database tables
    Base.metadata.create_all(bind=engine)

    # Create necessary directories
    config.create_directories()

    yield

    # Shutdown
    logger.info("Shutting down CASCADE Data Collector API")


# Create FastAPI app
app = FastAPI(
    title="CASCADE Data Collector API",
    description="KiwiSDR data collection for CASCADE neural network training",
    version="0.1.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(
    collectors.router,
    prefix="/collectors",
    tags=["collectors"],
)

app.include_router(
    processors.router,
    prefix="/processors",
    tags=["processors"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "CASCADE Data Collector",
        "version": "0.1.0",
        "status": "operational",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    # Check database connection
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        db_status = "disconnected"

    # Check Redis connection (if configured)
    redis_status = "not_configured"
    try:
        import redis
        r = redis.from_url(config.REDIS_URL)
        r.ping()
        redis_status = "connected"
    except Exception:
        redis_status = "disconnected"

    return {
        "status": "healthy" if db_status == "connected" else "degraded",
        "postgres": db_status,
        "redis": redis_status,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.DASHBOARD_HOST,
        port=config.DASHBOARD_PORT,
        reload=True,
    )