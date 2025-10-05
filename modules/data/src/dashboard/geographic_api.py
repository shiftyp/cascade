"""FastAPI backend for geographic diversity dashboard (T087).

Provides real-time diversity metrics and monitoring data.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncio
import random

from ..collectors.geographic_quotas import GeographicQuotaManager
from ..validators.geographic_diversity import GeographicDiversityValidator
from ..collectors.southern_priority import SouthernHemispherePriorityCollector

app = FastAPI(title="Geographic Diversity API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://cascade-collector.fly.dev", 
        "https://cascade-collector.fly.dev:3000",
        "http://cascade-collector.internal:3000",
        "https://cascade-collector.internal:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
quota_manager = GeographicQuotaManager()
diversity_validator = GeographicDiversityValidator(quota_manager)
southern_collector = SouthernHemispherePriorityCollector()


@app.get("/health")
async def health_check():
    """Health check endpoint for Fly.io and monitoring."""
    return {
        "status": "healthy",
        "service": "geographic_diversity_api",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/api/diversity/geographic-data")
async def get_geographic_data() -> Dict[str, Any]:
    """Get geographic distribution data for heatmap and progress bars."""
    progress = quota_manager.get_collection_progress()

    # Generate sample grid square data for visualization
    # In production, this would come from actual collection records
    grid_square_data = []
    for record in quota_manager.collection_history[-100:]:  # Last 100 records
        lat = quota_manager.classifier.get_latitude_from_grid(record["grid_square"])
        # Approximate longitude (simplified)
        lon = random.uniform(-180, 180)

        grid_square_data.append({
            "grid": record["grid_square"],
            "latitude": lat,
            "longitude": lon,
            "hours": record["hours"],
            "intensity": min(1.0, record["hours"] / 10)
        })

    return {
        "totalHours": progress.total_hours,
        "latitudeBands": {
            "arctic": {
                "hours": progress.latitude_band_hours.get("arctic", 0),
                "percentage": progress.latitude_band_percentages.get("arctic", 0),
                "target": 20
            },
            "temperate": {
                "hours": progress.latitude_band_hours.get("temperate", 0),
                "percentage": progress.latitude_band_percentages.get("temperate", 0),
                "target": 20
            },
            "tropical": {
                "hours": progress.latitude_band_hours.get("tropical", 0),
                "percentage": progress.latitude_band_percentages.get("tropical", 0),
                "target": 20
            },
            "antarctic": {
                "hours": progress.latitude_band_hours.get("antarctic", 0),
                "percentage": progress.latitude_band_percentages.get("antarctic", 0),
                "target": 20
            }
        },
        "hemispheres": {
            "north": {
                "hours": progress.hemisphere_hours.get("north", 0),
                "percentage": progress.hemisphere_percentages.get("north", 0),
                "target": 40
            },
            "south": {
                "hours": progress.hemisphere_hours.get("south", 0),
                "percentage": progress.hemisphere_percentages.get("south", 0),
                "target": 40
            },
            "equatorial": {
                "hours": progress.hemisphere_hours.get("equatorial", 0),
                "percentage": progress.hemisphere_percentages.get("equatorial", 0),
                "target": 20
            }
        },
        "gridSquareData": grid_square_data,
        "oceanPathPercentage": progress.ocean_path_percentage
    }


@app.get("/api/diversity/metrics")
async def get_diversity_metrics() -> Dict[str, Any]:
    """Get diversity metrics including Simpson's index and continental coverage."""
    metrics = diversity_validator.get_diversity_metrics()

    return {
        "simpsonDiversityIndex": metrics.simpson_diversity_index,
        "hemisphereBalanceScore": metrics.hemisphere_balance_score,
        "continentalCoverage": {
            "northAmerica": metrics.continental_coverage.get("north_america", False),
            "southAmerica": metrics.continental_coverage.get("south_america", False),
            "europe": metrics.continental_coverage.get("europe", False),
            "africa": metrics.continental_coverage.get("africa", False),
            "asia": metrics.continental_coverage.get("asia", False),
            "oceania": metrics.continental_coverage.get("oceania", False),
            "antarctica": metrics.continental_coverage.get("antarctica", False)
        },
        "overallDiversityScore": metrics.overall_diversity_score
    }


@app.get("/api/diversity/warnings")
async def get_warnings() -> Dict[str, List[str]]:
    """Get bias warnings when regions fall below 50% of target (T087c)."""
    warnings = quota_manager.get_quota_warnings()
    return {"warnings": warnings}


@app.get("/api/diversity/recommendations")
async def get_recommendations() -> Dict[str, List[Dict]]:
    """Get automatic rebalancing recommendations (T087d)."""
    recommendations = quota_manager.get_rebalancing_recommendations()

    # Format for frontend
    formatted = []
    for rec in recommendations:
        formatted.append({
            "region": rec["region"],
            "currentPercentage": rec["current_percentage"],
            "targetPercentage": rec["target_percentage"],
            "deficit": rec["deficit"],
            "priorityMultiplier": rec["priority_multiplier"],
            "action": rec["action"]
        })

    return {"recommendations": formatted}


@app.post("/api/diversity/add-sample")
async def add_collection_sample(
    grid_square: str,
    hours: float,
    is_ocean_path: bool = False
) -> Dict[str, str]:
    """Add a collection sample (for testing)."""
    quota_manager.add_collection_record(grid_square, hours, is_ocean_path)
    return {"status": "success", "message": f"Added {hours} hours for {grid_square}"}


@app.get("/api/diversity/southern-status")
async def get_southern_status() -> Dict[str, Any]:
    """Get southern hemisphere collection status."""
    status = southern_collector.get_southern_collection_status()
    return status


@app.get("/api/diversity/report")
async def get_diversity_report() -> Dict[str, str]:
    """Get human-readable diversity report."""
    report = diversity_validator.generate_diversity_report()
    return {"report": report}


@app.websocket("/ws/diversity-updates")
async def websocket_diversity_updates(websocket):
    """WebSocket endpoint for real-time diversity updates."""
    await websocket.accept()

    try:
        while True:
            # Send updates every 5 seconds
            await asyncio.sleep(5)

            # Get current data
            geographic_data = await get_geographic_data()
            metrics = await get_diversity_metrics()
            warnings = await get_warnings()

            update = {
                "timestamp": datetime.utcnow().isoformat(),
                "geographic_data": geographic_data,
                "metrics": metrics,
                "warnings": warnings["warnings"]
            }

            await websocket.send_json(update)

    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)