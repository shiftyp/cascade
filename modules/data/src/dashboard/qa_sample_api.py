"""
QA Sample Search API Endpoints for CASCADE Dashboard (T051d)

Provides REST API for searching and retrieving QA samples stored in S3.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict
import numpy as np
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncpg
import aioboto3
from contextlib import asynccontextmanager
import soundfile as sf
import io

# Import waterfall generator and IQ reader
from .waterfall_generator import WaterfallGenerator
from .iq_reader import IQStreamReader as IQReader

app = FastAPI(title="QA Sample Search API", version="1.0.0")

# CORS for Next.js dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://cascade-kiwi-collector.fly.dev"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
TIGRIS_ACCESS_KEY = os.getenv("TIGRIS_ACCESS_KEY")
TIGRIS_SECRET_KEY = os.getenv("TIGRIS_SECRET_KEY")
TIGRIS_BUCKET = os.getenv("TIGRIS_BUCKET", "cascade-iq-data")
TIGRIS_ENDPOINT = os.getenv("TIGRIS_ENDPOINT", "https://fly.storage.tigris.dev")

# Initialize components
waterfall_generator = WaterfallGenerator()
iq_reader = IQReader()

@dataclass
class QASample:
    """QA Sample data model matching TypeScript interface."""
    id: str
    session_id: str
    timestamp: datetime
    frequency_khz: float
    band: str
    mode: Optional[str] = None
    sample_rate: int = 12000
    duration: float = 1.0
    grid_square: Optional[str] = None
    callsign_hash: Optional[str] = None
    correlation_id: Optional[str] = None
    snr: Optional[float] = None
    propagation_mode: Optional[str] = None
    space_weather_condition: Optional[str] = None
    file_size_bytes: int = 0
    has_waterfall: bool = True
    s3_path: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON response."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['frequencyKhz'] = self.frequency_khz
        data['sessionId'] = self.session_id
        data['gridSquare'] = self.grid_square
        data['callsignHash'] = self.callsign_hash
        data['correlationId'] = self.correlation_id
        data['propagationMode'] = self.propagation_mode
        data['spaceWeatherCondition'] = self.space_weather_condition
        data['fileSizeBytes'] = self.file_size_bytes
        data['hasWaterfall'] = self.has_waterfall
        data['sampleRate'] = self.sample_rate
        # Remove snake_case versions
        for key in ['session_id', 'frequency_khz', 'grid_square', 'callsign_hash',
                   'correlation_id', 'propagation_mode', 'space_weather_condition',
                   'file_size_bytes', 'has_waterfall', 'sample_rate', 's3_path']:
            data.pop(key, None)
        return data

class SearchRequest(BaseModel):
    """Search request model for QA samples."""
    band: Optional[str] = Field(None, description="HF band (e.g., '20m', '40m')")
    callsign_hash: Optional[str] = Field(None, alias="callsignHash")
    start_date: Optional[datetime] = Field(None, alias="startDate")
    end_date: Optional[datetime] = Field(None, alias="endDate")
    propagation_mode: Optional[str] = Field(None, alias="propagationMode")
    min_snr: Optional[float] = Field(None, alias="minSNR", ge=-30, le=30)
    grid_square: Optional[str] = Field(None, alias="gridSquare")
    limit: int = Field(100, ge=1, le=1000)
    offset: int = Field(0, ge=0)

class WaterfallRequest(BaseModel):
    """Request model for waterfall generation."""
    recording_id: str = Field(..., description="Recording/session ID")
    start_time: float = Field(0, ge=0, description="Start time in seconds")
    duration: float = Field(10, gt=0, le=60, description="Duration in seconds")
    fft_size: int = Field(1024, description="FFT size for spectrogram")
    overlap: float = Field(0.5, ge=0, lt=1, description="Overlap ratio")
    colormap: str = Field("viridis", description="Colormap name")

@asynccontextmanager
async def get_db_connection():
    """Get database connection."""
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        yield conn
    finally:
        await conn.close()

@asynccontextmanager
async def get_s3_client():
    """Get S3 client for Tigris."""
    session = aioboto3.Session()
    async with session.client(
        's3',
        endpoint_url=TIGRIS_ENDPOINT,
        aws_access_key_id=TIGRIS_ACCESS_KEY,
        aws_secret_access_key=TIGRIS_SECRET_KEY,
        region_name='auto'
    ) as s3:
        yield s3

@app.get("/api/qa/search")
async def search_qa_samples(
    band: Optional[str] = Query(None),
    callsignHash: Optional[str] = Query(None),
    startDate: Optional[str] = Query(None),
    endDate: Optional[str] = Query(None),
    propagationMode: Optional[str] = Query(None),
    minSNR: Optional[float] = Query(None),
    gridSquare: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0)
) -> Dict[str, Any]:
    """
    Search QA samples based on filters.

    Returns paginated list of QA samples matching the criteria.
    """
    # Build SQL query
    query_parts = ["SELECT * FROM qa_samples WHERE 1=1"]
    params = []
    param_count = 0

    if band:
        param_count += 1
        query_parts.append(f" AND band = ${param_count}")
        params.append(band)

    if callsignHash:
        param_count += 1
        query_parts.append(f" AND callsign_hash LIKE ${param_count}")
        params.append(f"{callsignHash}%")

    if startDate:
        param_count += 1
        try:
            start_dt = datetime.fromisoformat(startDate.replace('Z', '+00:00'))
            query_parts.append(f" AND timestamp >= ${param_count}")
            params.append(start_dt)
        except:
            pass

    if endDate:
        param_count += 1
        try:
            end_dt = datetime.fromisoformat(endDate.replace('Z', '+00:00'))
            query_parts.append(f" AND timestamp <= ${param_count}")
            params.append(end_dt)
        except:
            pass

    if propagationMode:
        param_count += 1
        query_parts.append(f" AND propagation_mode = ${param_count}")
        params.append(propagationMode)

    if minSNR is not None:
        param_count += 1
        query_parts.append(f" AND snr >= ${param_count}")
        params.append(minSNR)

    if gridSquare:
        param_count += 1
        query_parts.append(f" AND grid_square = ${param_count}")
        params.append(gridSquare)

    # Add ordering and pagination
    query_parts.append(" ORDER BY timestamp DESC")
    param_count += 1
    query_parts.append(f" LIMIT ${param_count}")
    params.append(limit)
    param_count += 1
    query_parts.append(f" OFFSET ${param_count}")
    params.append(offset)

    query = "".join(query_parts)

    # Execute query
    samples = []
    total_count = 0

    async with get_db_connection() as conn:
        # Get total count
        count_query = query.replace("SELECT *", "SELECT COUNT(*)").split("ORDER BY")[0]
        count_params = params[:-2]  # Remove limit and offset
        total_count = await conn.fetchval(count_query, *count_params) or 0

        # Get samples
        rows = await conn.fetch(query, *params)

        for row in rows:
            sample = QASample(
                id=row['id'],
                session_id=row['session_id'],
                timestamp=row['timestamp'],
                frequency_khz=row['frequency_khz'],
                band=row['band'],
                mode=row.get('mode'),
                sample_rate=row.get('sample_rate', 12000),
                duration=row.get('duration', 1.0),
                grid_square=row.get('grid_square'),
                callsign_hash=row.get('callsign_hash'),
                correlation_id=row.get('correlation_id'),
                snr=row.get('snr'),
                propagation_mode=row.get('propagation_mode'),
                space_weather_condition=row.get('space_weather_condition'),
                file_size_bytes=row.get('file_size_bytes', 0),
                has_waterfall=row.get('has_waterfall', True),
                s3_path=row.get('s3_path'),
                metadata=row.get('metadata', {})
            )
            samples.append(sample.to_dict())

    return {
        "samples": samples,
        "total": total_count,
        "limit": limit,
        "offset": offset
    }

@app.post("/api/waterfall")
async def generate_waterfall(request: WaterfallRequest) -> Dict[str, Any]:
    """
    Generate waterfall data from IQ sample.

    Returns waterfall data, frequencies, and timestamps.
    """
    try:
        # Get S3 path for the recording
        async with get_db_connection() as conn:
            s3_path = await conn.fetchval(
                "SELECT s3_path FROM qa_samples WHERE session_id = $1",
                request.recording_id
            )

            if not s3_path:
                raise HTTPException(status_code=404, detail="Recording not found")

        # Download IQ data from S3
        async with get_s3_client() as s3:
            response = await s3.get_object(Bucket=TIGRIS_BUCKET, Key=s3_path)
            iq_data_bytes = await response['Body'].read()

        # Read IQ data
        iq_data = await iq_reader.read_from_bytes(
            iq_data_bytes,
            start_time=request.start_time,
            duration=request.duration
        )

        # Generate waterfall
        waterfall_data = waterfall_generator.generate(
            iq_data,
            sample_rate=12000,  # Standard KiwiSDR rate
            fft_size=request.fft_size,
            overlap=request.overlap,
            colormap=request.colormap
        )

        return {
            "waterfall": {
                "data": waterfall_data["normalized_data"].tolist(),
                "frequencies": waterfall_data["frequencies"].tolist(),
                "timestamps": waterfall_data["timestamps"].tolist(),
                "min_value": float(waterfall_data["min_value"]),
                "max_value": float(waterfall_data["max_value"])
            },
            "recording_id": request.recording_id,
            "duration": request.duration,
            "fft_size": request.fft_size
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/qa/{sample_id}")
async def get_qa_sample(sample_id: str) -> Dict[str, Any]:
    """Get a specific QA sample by ID."""
    async with get_db_connection() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM qa_samples WHERE id = $1",
            sample_id
        )

        if not row:
            raise HTTPException(status_code=404, detail="Sample not found")

        sample = QASample(
            id=row['id'],
            session_id=row['session_id'],
            timestamp=row['timestamp'],
            frequency_khz=row['frequency_khz'],
            band=row['band'],
            mode=row.get('mode'),
            sample_rate=row.get('sample_rate', 12000),
            duration=row.get('duration', 1.0),
            grid_square=row.get('grid_square'),
            callsign_hash=row.get('callsign_hash'),
            correlation_id=row.get('correlation_id'),
            snr=row.get('snr'),
            propagation_mode=row.get('propagation_mode'),
            space_weather_condition=row.get('space_weather_condition'),
            file_size_bytes=row.get('file_size_bytes', 0),
            has_waterfall=row.get('has_waterfall', True),
            s3_path=row.get('s3_path'),
            metadata=row.get('metadata', {})
        )

        return sample.to_dict()

@app.get("/api/qa/stats/summary")
async def get_qa_stats_summary() -> Dict[str, Any]:
    """Get summary statistics for QA samples."""
    async with get_db_connection() as conn:
        # Get overall stats
        stats = await conn.fetchrow("""
            SELECT
                COUNT(*) as total_samples,
                COUNT(DISTINCT band) as unique_bands,
                COUNT(DISTINCT grid_square) as unique_grids,
                COUNT(DISTINCT DATE(timestamp)) as collection_days,
                SUM(duration) as total_duration_seconds,
                SUM(file_size_bytes) as total_size_bytes,
                AVG(snr) as avg_snr,
                MIN(timestamp) as earliest_sample,
                MAX(timestamp) as latest_sample
            FROM qa_samples
        """)

        # Get band distribution
        band_dist = await conn.fetch("""
            SELECT band, COUNT(*) as count, SUM(duration) as total_duration
            FROM qa_samples
            GROUP BY band
            ORDER BY count DESC
        """)

        # Get propagation mode distribution
        prop_dist = await conn.fetch("""
            SELECT propagation_mode, COUNT(*) as count
            FROM qa_samples
            WHERE propagation_mode IS NOT NULL
            GROUP BY propagation_mode
            ORDER BY count DESC
        """)

        return {
            "total_samples": stats['total_samples'] or 0,
            "unique_bands": stats['unique_bands'] or 0,
            "unique_grids": stats['unique_grids'] or 0,
            "collection_days": stats['collection_days'] or 0,
            "total_duration_hours": (stats['total_duration_seconds'] or 0) / 3600,
            "total_size_gb": (stats['total_size_bytes'] or 0) / (1024**3),
            "average_snr": float(stats['avg_snr']) if stats['avg_snr'] else None,
            "date_range": {
                "start": stats['earliest_sample'].isoformat() if stats['earliest_sample'] else None,
                "end": stats['latest_sample'].isoformat() if stats['latest_sample'] else None
            },
            "band_distribution": [
                {
                    "band": row['band'],
                    "count": row['count'],
                    "duration_hours": row['total_duration'] / 3600
                }
                for row in band_dist
            ],
            "propagation_modes": [
                {"mode": row['propagation_mode'], "count": row['count']}
                for row in prop_dist
            ]
        }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "qa_sample_api"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)