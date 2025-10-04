"""Collector API endpoints.

Implements T039-T042: Collector API endpoints.
"""

import logging
from typing import Optional, Dict, Any, List
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..collectors.recorder import Recorder
from ..collectors.kiwi_client import KiwiClient
from ..models import get_db, RecordingSession, KiwiSDRSource

logger = logging.getLogger(__name__)
router = APIRouter()

# Global recorder instance
recorder = Recorder()


class ConnectRequest(BaseModel):
    """Connection request model."""

    url: str = Field(..., description="KiwiSDR URL (host:port)")
    frequency_khz: float = Field(..., description="Center frequency in kHz")
    mode: str = Field("iq", description="Recording mode")
    bandwidth_khz: float = Field(12.0, description="Bandwidth in kHz")
    auth: Optional[Dict[str, str]] = Field(None, description="Authentication credentials")


class ConnectResponse(BaseModel):
    """Connection response model."""

    session_id: str
    status: str
    kiwisdr_url: str
    frequency_khz: float


class StartRequest(BaseModel):
    """Start recording request."""

    session_id: Optional[str] = Field(None, description="Session ID")
    duration_seconds: int = Field(..., gt=0, description="Recording duration")
    output_format: str = Field("flac", description="Output format")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")
    start_at: Optional[datetime] = Field(None, description="Scheduled start time")


class StopRequest(BaseModel):
    """Stop recording request."""

    session_id: str = Field(..., description="Session ID to stop")
    save_to_tigris: bool = Field(True, description="Upload to Tigris storage")
    compress: bool = Field(True, description="Apply FLAC compression")
    delete_local: bool = Field(False, description="Delete local file after upload")
    force: bool = Field(False, description="Force stop")


@router.post("/connect", response_model=ConnectResponse)
async def connect_to_kiwisdr(
    request: ConnectRequest,
    db: Session = Depends(get_db),
):
    """POST /collectors/connect - Connect to a KiwiSDR (T039)."""
    try:
        # Create KiwiClient
        client = KiwiClient(request.url)

        # Connect
        success = await client.connect(
            frequency_khz=request.frequency_khz,
            mode=request.mode,
            bandwidth_khz=request.bandwidth_khz,
            auth=request.auth,
        )

        if not success:
            raise HTTPException(status_code=400, detail="Failed to connect to KiwiSDR")

        # Create session record
        session_id = client.session_id or str(datetime.utcnow().timestamp())

        return ConnectResponse(
            session_id=session_id,
            status="connected",
            kiwisdr_url=request.url,
            frequency_khz=request.frequency_khz,
        )

    except TimeoutError:
        raise HTTPException(status_code=504, detail="Connection timeout")
    except Exception as e:
        logger.error(f"Connection error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/start")
async def start_recording(
    request: StartRequest,
    db: Session = Depends(get_db),
):
    """POST /collectors/start - Start recording (T040)."""
    try:
        # Check if scheduled
        if request.start_at and request.start_at > datetime.utcnow():
            # TODO: Implement scheduling
            return {
                "status": "scheduled",
                "session_id": request.session_id,
                "start_at": request.start_at,
            }

        # Get metadata
        metadata = request.metadata or {}
        band = metadata.get("band")
        kiwisdr_url = metadata.get("kiwisdr_source", "default.kiwisdr.com:8073")

        # Start recording
        session_id = await recorder.start_recording(
            kiwisdr_url=kiwisdr_url,
            frequency_khz=14074,  # Default FT8 frequency
            duration_seconds=request.duration_seconds,
            band=band,
            session_id=request.session_id,
        )

        return {
            "status": "recording",
            "session_id": session_id,
            "start_time": datetime.utcnow(),
            "duration_seconds": request.duration_seconds,
        }

    except ConnectionError as e:
        raise HTTPException(status_code=409, detail=f"Not connected: {e}")
    except Exception as e:
        logger.error(f"Start recording error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/stop")
async def stop_recording(
    request: StopRequest,
    db: Session = Depends(get_db),
):
    """POST /collectors/stop - Stop recording (T041)."""
    try:
        # Stop recording
        result = await recorder.stop_recording(request.session_id)

        # Handle save options
        if request.save_to_tigris:
            # TODO: Implement Tigris upload
            result["saved_to_tigris"] = False

        result["compressed"] = request.compress

        return result

    except ValueError as e:
        raise HTTPException(status_code=404, detail=f"Session not found: {e}")
    except Exception as e:
        logger.error(f"Stop recording error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/status")
async def get_all_status(
    band: Optional[str] = Query(None, description="Filter by band"),
    status: Optional[str] = Query(None, description="Filter by status"),
    db: Session = Depends(get_db),
):
    """GET /collectors/status - Get status of all collectors (T042)."""
    try:
        # Query sessions
        query = db.query(RecordingSession)

        if band:
            query = query.filter(RecordingSession.band == band)
        if status:
            query = query.filter(RecordingSession.status == status)

        sessions = query.all()

        # Get active sessions from recorder
        active_sessions = recorder.get_active_sessions()

        return {
            "collectors": [
                {
                    "session_id": str(s.session_id),
                    "status": s.status,
                    "frequency_khz": s.frequency_khz,
                    "band": s.band,
                    "start_time": s.start_time,
                    "duration_seconds": s.duration_seconds,
                    "bytes_recorded": s.file_size_bytes,
                }
                for s in sessions
            ],
            "total_active": len(active_sessions),
            "total_sdrs": db.query(KiwiSDRSource).filter_by(active=True).count(),
        }

    except Exception as e:
        logger.error(f"Status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{session_id}")
async def get_session_status(
    session_id: str,
    db: Session = Depends(get_db),
):
    """GET /collectors/status/{session_id} - Get single session status."""
    session = (
        db.query(RecordingSession)
        .filter(RecordingSession.session_id == session_id)
        .first()
    )

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    return {
        "session_id": str(session.session_id),
        "status": session.status,
        "frequency_khz": session.frequency_khz,
        "band": session.band,
        "duration_seconds": session.duration_seconds,
        "bytes_recorded": session.file_size_bytes,
        "start_time": session.start_time,
        "end_time": session.end_time,
    }


@router.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """GET /collectors/health - Health check."""
    try:
        # Check database
        db.execute("SELECT 1")
        postgres_connected = True
    except:
        postgres_connected = False

    # Check Redis
    redis_connected = False
    try:
        import redis
        from src.config import config

        r = redis.from_url(config.REDIS_URL)
        r.ping()
        redis_connected = True
    except:
        pass

    healthy = postgres_connected

    return {
        "healthy": healthy,
        "postgres_connected": postgres_connected,
        "redis_connected": redis_connected,
        "worker_count": len(recorder.active_sessions),
    }


@router.get("/metrics")
async def get_metrics(db: Session = Depends(get_db)):
    """GET /collectors/metrics - Get collection metrics."""
    from sqlalchemy import func

    # Calculate metrics
    total_sessions = db.query(RecordingSession).count()
    completed_sessions = (
        db.query(RecordingSession)
        .filter(RecordingSession.status == "completed")
        .count()
    )

    # Total hours collected
    total_seconds = (
        db.query(func.sum(RecordingSession.duration_seconds))
        .filter(RecordingSession.status == "completed")
        .scalar()
        or 0
    )
    total_hours = total_seconds / 3600

    # Hours today
    from datetime import date

    today_seconds = (
        db.query(func.sum(RecordingSession.duration_seconds))
        .filter(
            RecordingSession.status == "completed",
            func.date(RecordingSession.created_at) == date.today(),
        )
        .scalar()
        or 0
    )
    hours_today = today_seconds / 3600

    # Storage used
    total_bytes = (
        db.query(func.sum(RecordingSession.file_size_bytes))
        .filter(RecordingSession.file_size_bytes.isnot(None))
        .scalar()
        or 0
    )
    storage_gb = total_bytes / (1024**3)

    # Active SDRs
    active_sdrs = (
        db.query(KiwiSDRSource)
        .filter(KiwiSDRSource.active == True)
        .count()
    )

    return {
        "total_hours_collected": round(total_hours, 2),
        "hours_today": round(hours_today, 2),
        "active_sdrs": active_sdrs,
        "storage_used_gb": round(storage_gb, 2),
        "collection_rate_per_hour": round(hours_today / 24, 2) if hours_today else 0,
        "total_sessions": total_sessions,
        "completed_sessions": completed_sessions,
    }