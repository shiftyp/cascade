"""
QA Metadata Aggregator for CASCADE Data Collection (T051e)

Aggregates metadata from 1% QA samples for dashboard display and analysis.
"""

import asyncio
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, asdict, field
import asyncpg
import numpy as np
from collections import defaultdict
import hashlib

@dataclass
class QAMetadata:
    """Metadata for a QA sample."""
    sample_id: str
    session_id: str
    timestamp: datetime
    frequency_khz: float
    band: str
    sample_rate: int
    duration: float
    grid_square: Optional[str] = None
    callsign_hash: Optional[str] = None

    # Signal characteristics
    snr: Optional[float] = None
    signal_type: Optional[str] = None  # FT8, WSPR, CW, SSB, QRN
    occupied_bandwidth_hz: Optional[float] = None
    peak_frequency_offset_hz: Optional[float] = None

    # Propagation information
    propagation_mode: Optional[str] = None  # F2, Es, TEP, EME, MS
    estimated_distance_km: Optional[float] = None
    estimated_hop_count: Optional[int] = None
    path_type: Optional[str] = None  # ocean, land, mixed

    # Space weather context
    space_weather_condition: Optional[str] = None
    solar_flux: Optional[float] = None
    k_index: Optional[int] = None
    a_index: Optional[int] = None

    # File information
    file_size_bytes: int = 0
    compression_ratio: Optional[float] = None
    s3_path: Optional[str] = None
    has_waterfall: bool = True

    # Quality metrics
    quality_score: Optional[float] = None
    clipping_detected: bool = False
    noise_floor_db: Optional[float] = None

    # Processing metadata
    processing_version: str = "1.0.0"
    extraction_timestamp: Optional[datetime] = None
    correlation_id: Optional[str] = None

@dataclass
class AggregatedStats:
    """Aggregated statistics for QA samples."""
    total_samples: int = 0
    total_duration_hours: float = 0.0
    total_size_gb: float = 0.0

    # Band distribution
    band_hours: Dict[str, float] = field(default_factory=dict)
    band_counts: Dict[str, int] = field(default_factory=dict)

    # Geographic distribution
    grid_square_counts: Dict[str, int] = field(default_factory=dict)
    hemisphere_distribution: Dict[str, float] = field(default_factory=lambda: {
        "north": 0.0, "south": 0.0, "equatorial": 0.0
    })

    # Signal type distribution
    signal_types: Dict[str, int] = field(default_factory=dict)
    propagation_modes: Dict[str, int] = field(default_factory=dict)

    # Quality metrics
    average_snr: Optional[float] = None
    average_quality_score: Optional[float] = None
    clipping_percentage: float = 0.0

    # Time distribution
    hourly_distribution: List[float] = field(default_factory=lambda: [0.0] * 24)
    daily_trend: List[Tuple[str, float]] = field(default_factory=list)

    # Space weather correlation
    space_weather_distribution: Dict[str, int] = field(default_factory=dict)
    high_k_index_samples: int = 0

class QAMetadataAggregator:
    """Aggregates QA sample metadata for analysis and visualization."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self._pool: Optional[asyncpg.Pool] = None
        self._cache: Dict[str, Any] = {}
        self._cache_ttl = 300  # 5 minutes
        self._last_cache_update = datetime.now(timezone.utc)

    async def initialize(self):
        """Initialize database connection pool."""
        self._pool = await asyncpg.create_pool(
            self.database_url,
            min_size=2,
            max_size=10,
            command_timeout=60
        )

        # Ensure QA samples table exists
        async with self._pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS qa_samples (
                    id VARCHAR(64) PRIMARY KEY,
                    session_id VARCHAR(64) NOT NULL,
                    timestamp TIMESTAMPTZ NOT NULL,
                    frequency_khz REAL NOT NULL,
                    band VARCHAR(10) NOT NULL,
                    sample_rate INTEGER DEFAULT 12000,
                    duration REAL DEFAULT 1.0,
                    grid_square VARCHAR(6),
                    callsign_hash VARCHAR(16),
                    snr REAL,
                    signal_type VARCHAR(20),
                    occupied_bandwidth_hz REAL,
                    peak_frequency_offset_hz REAL,
                    propagation_mode VARCHAR(20),
                    estimated_distance_km REAL,
                    estimated_hop_count INTEGER,
                    path_type VARCHAR(20),
                    space_weather_condition VARCHAR(20),
                    solar_flux REAL,
                    k_index INTEGER,
                    a_index INTEGER,
                    file_size_bytes BIGINT DEFAULT 0,
                    compression_ratio REAL,
                    s3_path TEXT,
                    has_waterfall BOOLEAN DEFAULT TRUE,
                    quality_score REAL,
                    clipping_detected BOOLEAN DEFAULT FALSE,
                    noise_floor_db REAL,
                    processing_version VARCHAR(20) DEFAULT '1.0.0',
                    extraction_timestamp TIMESTAMPTZ,
                    correlation_id VARCHAR(64),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),

                    INDEX idx_qa_timestamp (timestamp),
                    INDEX idx_qa_band (band),
                    INDEX idx_qa_grid (grid_square),
                    INDEX idx_qa_session (session_id),
                    INDEX idx_qa_signal_type (signal_type)
                )
            """)

    async def close(self):
        """Close database connection pool."""
        if self._pool:
            await self._pool.close()

    async def ingest_qa_sample(self, metadata: QAMetadata) -> str:
        """
        Ingest a new QA sample into the database.

        Returns sample ID.
        """
        if not metadata.sample_id:
            # Generate deterministic ID
            hash_input = f"{metadata.session_id}_{metadata.timestamp.isoformat()}_{metadata.frequency_khz}"
            metadata.sample_id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]

        async with self._pool.acquire() as conn:
            # Upsert sample
            await conn.execute("""
                INSERT INTO qa_samples (
                    id, session_id, timestamp, frequency_khz, band,
                    sample_rate, duration, grid_square, callsign_hash,
                    snr, signal_type, occupied_bandwidth_hz, peak_frequency_offset_hz,
                    propagation_mode, estimated_distance_km, estimated_hop_count, path_type,
                    space_weather_condition, solar_flux, k_index, a_index,
                    file_size_bytes, compression_ratio, s3_path, has_waterfall,
                    quality_score, clipping_detected, noise_floor_db,
                    processing_version, extraction_timestamp, correlation_id,
                    metadata
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                    $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
                    $31, $32
                )
                ON CONFLICT (id) DO UPDATE SET
                    snr = EXCLUDED.snr,
                    signal_type = EXCLUDED.signal_type,
                    propagation_mode = EXCLUDED.propagation_mode,
                    space_weather_condition = EXCLUDED.space_weather_condition,
                    quality_score = EXCLUDED.quality_score,
                    metadata = EXCLUDED.metadata
            """,
                metadata.sample_id, metadata.session_id, metadata.timestamp,
                metadata.frequency_khz, metadata.band, metadata.sample_rate,
                metadata.duration, metadata.grid_square, metadata.callsign_hash,
                metadata.snr, metadata.signal_type, metadata.occupied_bandwidth_hz,
                metadata.peak_frequency_offset_hz, metadata.propagation_mode,
                metadata.estimated_distance_km, metadata.estimated_hop_count,
                metadata.path_type, metadata.space_weather_condition,
                metadata.solar_flux, metadata.k_index, metadata.a_index,
                metadata.file_size_bytes, metadata.compression_ratio,
                metadata.s3_path, metadata.has_waterfall, metadata.quality_score,
                metadata.clipping_detected, metadata.noise_floor_db,
                metadata.processing_version, metadata.extraction_timestamp,
                metadata.correlation_id,
                json.dumps(asdict(metadata)) if metadata else None
            )

        # Invalidate cache
        self._cache.clear()

        return metadata.sample_id

    async def aggregate_stats(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        band: Optional[str] = None
    ) -> AggregatedStats:
        """
        Aggregate statistics for QA samples.

        Args:
            start_date: Start of date range (default: 30 days ago)
            end_date: End of date range (default: now)
            band: Optional band filter

        Returns:
            Aggregated statistics
        """
        # Check cache
        cache_key = f"stats_{start_date}_{end_date}_{band}"
        if cache_key in self._cache:
            cache_time = self._cache.get(f"{cache_key}_time")
            if cache_time and (datetime.now(timezone.utc) - cache_time).seconds < self._cache_ttl:
                return self._cache[cache_key]

        # Default date range
        if not end_date:
            end_date = datetime.now(timezone.utc)
        if not start_date:
            start_date = end_date - timedelta(days=30)

        stats = AggregatedStats()

        async with self._pool.acquire() as conn:
            # Build query conditions
            conditions = ["timestamp >= $1 AND timestamp <= $2"]
            params = [start_date, end_date]

            if band:
                conditions.append(f"band = ${len(params) + 1}")
                params.append(band)

            where_clause = " AND ".join(conditions)

            # Get overall stats
            row = await conn.fetchrow(f"""
                SELECT
                    COUNT(*) as total,
                    SUM(duration) as total_duration,
                    SUM(file_size_bytes) as total_size,
                    AVG(snr) as avg_snr,
                    AVG(quality_score) as avg_quality,
                    SUM(CASE WHEN clipping_detected THEN 1 ELSE 0 END) as clipping_count
                FROM qa_samples
                WHERE {where_clause}
            """, *params)

            if row:
                stats.total_samples = row['total'] or 0
                stats.total_duration_hours = (row['total_duration'] or 0) / 3600
                stats.total_size_gb = (row['total_size'] or 0) / (1024**3)
                stats.average_snr = row['avg_snr']
                stats.average_quality_score = row['avg_quality']
                stats.clipping_percentage = (row['clipping_count'] or 0) / max(1, stats.total_samples) * 100

            # Get band distribution
            band_rows = await conn.fetch(f"""
                SELECT band, COUNT(*) as count, SUM(duration) as duration
                FROM qa_samples
                WHERE {where_clause}
                GROUP BY band
            """, *params)

            for row in band_rows:
                stats.band_counts[row['band']] = row['count']
                stats.band_hours[row['band']] = row['duration'] / 3600

            # Get grid square distribution
            grid_rows = await conn.fetch(f"""
                SELECT grid_square, COUNT(*) as count
                FROM qa_samples
                WHERE {where_clause} AND grid_square IS NOT NULL
                GROUP BY grid_square
                ORDER BY count DESC
                LIMIT 100
            """, *params)

            for row in grid_rows:
                stats.grid_square_counts[row['grid_square']] = row['count']

            # Calculate hemisphere distribution
            for grid_square, count in stats.grid_square_counts.items():
                if len(grid_square) >= 2:
                    lat_char = grid_square[1]
                    if lat_char >= 'K':  # Northern hemisphere
                        stats.hemisphere_distribution['north'] += count
                    elif lat_char <= 'G':  # Southern hemisphere
                        stats.hemisphere_distribution['south'] += count
                    else:  # Equatorial
                        stats.hemisphere_distribution['equatorial'] += count

            # Get signal type distribution
            signal_rows = await conn.fetch(f"""
                SELECT signal_type, COUNT(*) as count
                FROM qa_samples
                WHERE {where_clause} AND signal_type IS NOT NULL
                GROUP BY signal_type
            """, *params)

            for row in signal_rows:
                stats.signal_types[row['signal_type']] = row['count']

            # Get propagation mode distribution
            prop_rows = await conn.fetch(f"""
                SELECT propagation_mode, COUNT(*) as count
                FROM qa_samples
                WHERE {where_clause} AND propagation_mode IS NOT NULL
                GROUP BY propagation_mode
            """, *params)

            for row in prop_rows:
                stats.propagation_modes[row['propagation_mode']] = row['count']

            # Get hourly distribution
            hour_rows = await conn.fetch(f"""
                SELECT EXTRACT(HOUR FROM timestamp) as hour, COUNT(*) as count
                FROM qa_samples
                WHERE {where_clause}
                GROUP BY hour
                ORDER BY hour
            """, *params)

            for row in hour_rows:
                hour_idx = int(row['hour'])
                if 0 <= hour_idx < 24:
                    stats.hourly_distribution[hour_idx] = float(row['count'])

            # Get daily trend (last 30 days)
            trend_rows = await conn.fetch(f"""
                SELECT DATE(timestamp) as day, SUM(duration) as duration
                FROM qa_samples
                WHERE {where_clause}
                GROUP BY day
                ORDER BY day DESC
                LIMIT 30
            """, *params)

            stats.daily_trend = [
                (row['day'].isoformat(), row['duration'] / 3600)
                for row in trend_rows
            ]

            # Get space weather distribution
            weather_rows = await conn.fetch(f"""
                SELECT space_weather_condition, COUNT(*) as count
                FROM qa_samples
                WHERE {where_clause} AND space_weather_condition IS NOT NULL
                GROUP BY space_weather_condition
            """, *params)

            for row in weather_rows:
                stats.space_weather_distribution[row['space_weather_condition']] = row['count']

            # Count high K-index samples
            high_k = await conn.fetchval(f"""
                SELECT COUNT(*) FROM qa_samples
                WHERE {where_clause} AND k_index >= 5
            """, *params)
            stats.high_k_index_samples = high_k or 0

        # Cache results
        self._cache[cache_key] = stats
        self._cache[f"{cache_key}_time"] = datetime.now(timezone.utc)

        return stats

    async def find_similar_samples(
        self,
        reference_id: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Find samples similar to a reference sample.

        Uses frequency, band, SNR, and propagation mode for similarity.
        """
        async with self._pool.acquire() as conn:
            # Get reference sample
            ref = await conn.fetchrow(
                "SELECT * FROM qa_samples WHERE id = $1",
                reference_id
            )

            if not ref:
                return []

            # Find similar samples
            similar = await conn.fetch("""
                SELECT *,
                    ABS(frequency_khz - $2) as freq_diff,
                    ABS(COALESCE(snr, 0) - COALESCE($3, 0)) as snr_diff
                FROM qa_samples
                WHERE id != $1
                    AND band = $4
                    AND ABS(frequency_khz - $2) < 100  -- Within 100 kHz
                ORDER BY
                    CASE WHEN propagation_mode = $5 THEN 0 ELSE 1 END,
                    freq_diff,
                    snr_diff
                LIMIT $6
            """,
                reference_id, ref['frequency_khz'], ref['snr'],
                ref['band'], ref['propagation_mode'], limit
            )

            return [dict(row) for row in similar]

    async def get_quality_distribution(self) -> Dict[str, Any]:
        """Get distribution of quality scores across samples."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    CASE
                        WHEN quality_score >= 0.9 THEN 'excellent'
                        WHEN quality_score >= 0.7 THEN 'good'
                        WHEN quality_score >= 0.5 THEN 'fair'
                        WHEN quality_score >= 0.3 THEN 'poor'
                        ELSE 'very_poor'
                    END as quality_category,
                    COUNT(*) as count,
                    AVG(snr) as avg_snr
                FROM qa_samples
                WHERE quality_score IS NOT NULL
                GROUP BY quality_category
                ORDER BY
                    CASE quality_category
                        WHEN 'excellent' THEN 1
                        WHEN 'good' THEN 2
                        WHEN 'fair' THEN 3
                        WHEN 'poor' THEN 4
                        ELSE 5
                    END
            """)

            return {
                "categories": [
                    {
                        "category": row['quality_category'],
                        "count": row['count'],
                        "average_snr": float(row['avg_snr']) if row['avg_snr'] else None
                    }
                    for row in rows
                ]
            }

    async def export_metadata(
        self,
        output_format: str = "json",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> bytes:
        """
        Export metadata in specified format.

        Supports JSON and CSV formats.
        """
        # Default date range
        if not end_date:
            end_date = datetime.now(timezone.utc)
        if not start_date:
            start_date = end_date - timedelta(days=30)

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM qa_samples
                WHERE timestamp >= $1 AND timestamp <= $2
                ORDER BY timestamp DESC
            """, start_date, end_date)

            if output_format.lower() == "csv":
                # Generate CSV
                import csv
                import io

                output = io.StringIO()
                if rows:
                    writer = csv.DictWriter(output, fieldnames=rows[0].keys())
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(dict(row))

                return output.getvalue().encode('utf-8')

            else:  # JSON
                data = [dict(row) for row in rows]
                # Convert datetime objects to ISO strings
                for item in data:
                    for key, value in item.items():
                        if isinstance(value, datetime):
                            item[key] = value.isoformat()

                return json.dumps(data, indent=2).encode('utf-8')

# Usage example
async def main():
    """Example usage of QA metadata aggregator."""
    aggregator = QAMetadataAggregator(os.getenv("DATABASE_URL"))
    await aggregator.initialize()

    # Ingest a sample
    metadata = QAMetadata(
        sample_id="test_001",
        session_id="session_001",
        timestamp=datetime.now(timezone.utc),
        frequency_khz=14074.0,
        band="20m",
        duration=10.0,
        grid_square="FN42",
        snr=12.5,
        signal_type="FT8",
        propagation_mode="F2"
    )

    sample_id = await aggregator.ingest_qa_sample(metadata)
    print(f"Ingested sample: {sample_id}")

    # Get aggregated stats
    stats = await aggregator.aggregate_stats()
    print(f"Total samples: {stats.total_samples}")
    print(f"Total hours: {stats.total_duration_hours:.2f}")
    print(f"Band distribution: {stats.band_counts}")

    await aggregator.close()

if __name__ == "__main__":
    import os
    asyncio.run(main())