"""FastAPI web dashboard for CASCADE data collection with QA waterfall viewer."""

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncpg
import asyncio
import json
import logging
from pathlib import Path
import os

logger = logging.getLogger(__name__)

app = FastAPI(title="CASCADE Dashboard", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection pool
db_pool = None


async def get_db_pool():
    """Get or create database connection pool."""
    global db_pool
    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", 5432)),
            database=os.getenv("DB_NAME", "cascade"),
            user=os.getenv("DB_USER", "cascade"),
            password=os.getenv("DB_PASSWORD", ""),
            min_size=5,
            max_size=20
        )
    return db_pool


@app.on_event("startup")
async def startup_event():
    """Initialize database connection on startup."""
    await get_db_pool()


@app.on_event("shutdown")
async def shutdown_event():
    """Close database connections on shutdown."""
    global db_pool
    if db_pool:
        await db_pool.close()


# API Endpoints

@app.get("/api/status")
async def get_collection_status():
    """Get overall collection status."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow("SELECT * FROM v_collection_status")
        if row:
            return dict(row)
        return {}


@app.get("/api/hourly-stats")
async def get_hourly_stats(
    hours: int = Query(24, ge=1, le=168, description="Number of hours to retrieve")
):
    """Get hourly collection statistics."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT * FROM v_hourly_collection_stats
            WHERE hour_bin >= NOW() - INTERVAL '{hours} hours'
            ORDER BY hour_bin DESC
        """)
        return [dict(row) for row in rows]


@app.get("/api/sdr-performance")
async def get_sdr_performance(
    active_only: bool = Query(False, description="Show only active SDRs")
):
    """Get SDR performance metrics."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        query = """
            SELECT * FROM v_sdr_performance
            WHERE last_used >= NOW() - INTERVAL '24 hours'
        """
        if active_only:
            query += " AND is_active = true"
        query += " ORDER BY is_active DESC, total_hours DESC LIMIT 50"

        rows = await conn.fetch(query)
        return [dict(row) for row in rows]


@app.get("/api/band-coverage")
async def get_band_coverage():
    """Get band coverage statistics."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT frequency_band,
                   EXTRACT(HOUR FROM hour_slot) as hour,
                   avg_hours,
                   avg_sdrs
            FROM v_band_coverage
            ORDER BY frequency_band, hour
        """)

        # Transform to heatmap format
        coverage = {}
        for row in rows:
            band = row['frequency_band']
            if band not in coverage:
                coverage[band] = [0] * 24
            coverage[band][int(row['hour'])] = float(row['avg_hours'])

        return coverage


@app.get("/api/space-weather")
async def get_space_weather(
    days: int = Query(2, ge=1, le=7, description="Number of days to retrieve")
):
    """Get recent space weather events."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch(f"""
            SELECT * FROM v_space_weather_events
            WHERE timestamp >= NOW() - INTERVAL '{days} days'
            ORDER BY timestamp DESC
            LIMIT 100
        """)
        return [dict(row) for row in rows]


@app.get("/api/storage")
async def get_storage_usage():
    """Get storage usage breakdown."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        rows = await conn.fetch("SELECT * FROM v_storage_usage ORDER BY total_gb DESC")
        return [dict(row) for row in rows]


@app.get("/api/qa-samples")
async def get_qa_samples(
    band: Optional[str] = Query(None, description="Filter by frequency band"),
    min_quality: Optional[float] = Query(None, ge=0, le=100, description="Minimum quality score"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum results")
):
    """Get QA sample metadata."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        query = """
            SELECT
                qs.id,
                qs.recording_session_id,
                qs.timestamp,
                qs.frequency_band,
                qs.quality_score,
                qs.file_path,
                qs.is_quarantined,
                rs.kiwisdr_source_id,
                ks.name as sdr_name,
                ks.location_grid
            FROM qa_samples qs
            JOIN recording_sessions rs ON qs.recording_session_id = rs.id
            JOIN kiwisdr_sources ks ON rs.kiwisdr_source_id = ks.id
            WHERE 1=1
        """

        params = []
        param_count = 0

        if band:
            param_count += 1
            query += f" AND qs.frequency_band = ${param_count}"
            params.append(band)

        if min_quality is not None:
            param_count += 1
            query += f" AND qs.quality_score >= ${param_count}"
            params.append(min_quality)

        query += f" ORDER BY qs.timestamp DESC LIMIT ${param_count + 1}"
        params.append(limit)

        rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]


@app.get("/api/qa-sample/{sample_id}/waterfall")
async def get_qa_waterfall_data(sample_id: str):
    """Get waterfall data for a specific QA sample."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        # Get sample metadata
        sample = await conn.fetchrow("""
            SELECT * FROM qa_samples WHERE id = $1
        """, sample_id)

        if not sample:
            raise HTTPException(status_code=404, detail="Sample not found")

        # Generate waterfall data (placeholder - actual implementation in waterfall_generator.py)
        waterfall_data = {
            "sample_id": sample_id,
            "timestamp": sample['timestamp'].isoformat(),
            "frequency_band": sample['frequency_band'],
            "center_frequency": sample.get('center_frequency', 14100000),
            "sample_rate": 12000,
            "waterfall_url": f"/api/qa-sample/{sample_id}/waterfall-image",
            "iq_data_url": f"/api/qa-sample/{sample_id}/iq-data"
        }

        return waterfall_data


@app.get("/api/correlations")
async def get_correlations(
    complete_only: bool = Query(False, description="Show only complete correlations")
):
    """Get correlation completeness statistics."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        query = "SELECT * FROM v_correlation_completeness"
        if complete_only:
            query += " WHERE complete_pairs > 0"
        query += " ORDER BY collection_date DESC LIMIT 30"

        rows = await conn.fetch(query)
        return [dict(row) for row in rows]


@app.get("/api/propagation")
async def get_propagation_summary(
    band: Optional[str] = Query(None, description="Filter by frequency band"),
    mode: Optional[str] = Query(None, description="Filter by mode (FT8/WSPR)")
):
    """Get propagation data summary."""
    pool = await get_db_pool()
    async with pool.acquire() as conn:
        query = "SELECT * FROM v_propagation_summary WHERE 1=1"
        params = []
        param_count = 0

        if band:
            param_count += 1
            query += f" AND frequency_band = ${param_count}"
            params.append(band)

        if mode:
            param_count += 1
            query += f" AND mode = ${param_count}"
            params.append(mode)

        query += " ORDER BY total_records DESC"

        rows = await conn.fetch(query, *params)
        return [dict(row) for row in rows]


# WebSocket for real-time updates
@app.websocket("/ws/updates")
async def websocket_updates(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates."""
    await websocket.accept()

    try:
        while True:
            # Send updates every 5 seconds
            pool = await get_db_pool()
            async with pool.acquire() as conn:
                # Get current status
                status = await conn.fetchrow("SELECT * FROM v_collection_status")

                # Get active SDR count
                active_sdrs = await conn.fetchval("""
                    SELECT COUNT(*) FROM kiwisdr_sources
                    WHERE is_active = true
                """)

                # Get recent recordings count (last 5 minutes)
                recent_recordings = await conn.fetchval("""
                    SELECT COUNT(*) FROM recording_sessions
                    WHERE start_time >= NOW() - INTERVAL '5 minutes'
                """)

                update = {
                    "type": "status_update",
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": {
                        "total_hours": float(status['total_hours_collected']) if status else 0,
                        "active_sdrs": active_sdrs,
                        "recent_recordings": recent_recordings,
                        "storage_gb": float(status['total_storage_gb']) if status else 0
                    }
                }

                await websocket.send_json(update)

            await asyncio.sleep(5)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()


# HTML Dashboard Page
@app.get("/", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the main dashboard HTML page."""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CASCADE Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0a0e1a;
            color: #e0e6ed;
            line-height: 1.6;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1rem 2rem;
            box-shadow: 0 2px 10px rgba(0,0,0,0.3);
        }
        .header h1 {
            font-size: 1.8rem;
            font-weight: 300;
            letter-spacing: 2px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        .stat-card {
            background: #1a1f2e;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #2d3446;
        }
        .stat-card h3 {
            color: #7c8798;
            font-size: 0.9rem;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }
        .stat-value {
            font-size: 2rem;
            font-weight: 600;
            color: #fff;
        }
        .chart-container {
            background: #1a1f2e;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #2d3446;
            margin-bottom: 2rem;
        }
        .chart-container h2 {
            margin-bottom: 1rem;
            color: #fff;
        }
        #waterfall-viewer {
            height: 400px;
            background: #0a0e1a;
            border-radius: 4px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #7c8798;
        }
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 0.5rem;
        }
        .status-active { background: #4ade80; }
        .status-inactive { background: #f87171; }
        .loading {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>CASCADE DATA COLLECTION DASHBOARD</h1>
    </div>

    <div class="container">
        <div class="stats-grid">
            <div class="stat-card">
                <h3>Total Hours</h3>
                <div class="stat-value loading" id="total-hours">--</div>
            </div>
            <div class="stat-card">
                <h3>Active SDRs</h3>
                <div class="stat-value loading" id="active-sdrs">--</div>
            </div>
            <div class="stat-card">
                <h3>Storage Used</h3>
                <div class="stat-value loading" id="storage-used">--</div>
            </div>
            <div class="stat-card">
                <h3>Collection Rate</h3>
                <div class="stat-value loading" id="collection-rate">--</div>
            </div>
        </div>

        <div class="chart-container">
            <h2>24-Hour Collection Timeline</h2>
            <canvas id="timeline-chart" height="100"></canvas>
        </div>

        <div class="chart-container">
            <h2>QA Sample Waterfall Viewer</h2>
            <div id="waterfall-viewer">
                <p>Select a QA sample to view waterfall</p>
            </div>
        </div>

        <div class="chart-container">
            <h2>Band Coverage Heatmap</h2>
            <canvas id="band-heatmap" height="200"></canvas>
        </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="/static/dashboard.js"></script>
</body>
</html>
    """
    return HTMLResponse(content=html_content)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        pool = await get_db_pool()
        async with pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)