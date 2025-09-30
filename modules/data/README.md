# CASCADE Data Module - Developer Guide

The CASCADE Data Module collects real-world HF radio propagation data from the global KiwiSDR/WebSDR network for training neural network models. This is the foundational module in the CASCADE architecture.

## Quick Start

```bash
# Start all services with Docker Compose
docker-compose up -d

# Access services
# - Dashboard: http://localhost:3000
# - API: http://localhost:8000/api/diversity/metrics
# - PostgreSQL: http://localhost:8082 (Adminer)
# - Redis: http://localhost:8081 (Redis Commander)

# View logs
docker-compose logs -f scheduler worker

# Run tests
pytest tests/ -v
```

## Project Goals

- **Target**: 200,000-300,000 hours of HF radio data over 18 months (weighted collection)
- **Baseline**: 4,000 hours per HF band (24,000 hours minimum)
- **Sources**: 50-100 simultaneous KiwiSDR/WebSDR connections (scales to 200+ during events)
- **Storage**: 35-75TB compressed data (FLAC format, varies with seasonal/event weighting)
- **Privacy**: All callsigns anonymized via one-way hashing
- **Collection Rate**: 200-500 hrs/day (weighted average accounting for seasonal variance and event scaling)

## Architecture

### Module Structure

```
modules/data/
├── src/
│   ├── collectors/       # KiwiSDR/WebSDR connection and scheduling
│   ├── processors/       # FT8/WSPR decoding, QRN analysis, power estimation
│   ├── validators/       # Quality checks, geographic diversity metrics
│   ├── storage/          # FLAC compression, Tigris S3, PostgreSQL
│   ├── analytics/        # Station fingerprinting, pattern analysis
│   ├── embeddings/       # Station-aware embedding generation
│   ├── dashboard/        # Geographic diversity monitoring (FastAPI + Next.js)
│   ├── api/              # REST API endpoints
│   ├── models/           # SQLAlchemy database models
│   ├── external/         # NOAA space weather integration
│   └── notifications/    # Gmail alerts for collection issues
├── tests/
│   ├── unit/             # Component tests
│   ├── integration/      # End-to-end pipeline tests
│   └── contract/         # API contract tests
├── migrations/           # PostgreSQL schema migrations
├── scripts/              # Utility scripts
└── config/               # Configuration files
```

### Key Components

#### Collectors
- **kiwi_client.py**: KiwiSDR WebSocket connection handler
- **websdr_client.py**: WebSDR HTTP stream handler
- **scheduler.py**: Central coordinator for collection scheduling
- **worker.py**: Distributed worker process for parallel SDR connections
- **sdr_manager.py**: Propagation-aware SDR rotation (respects 30-90min limits)
- **hybrid_sdr_selector.py**: Chooses optimal KiwiSDR vs WebSDR based on availability
- **event_scaler.py**: Scales to 100-200 SDRs during solar/propagation events
- **geographic_quotas.py**: Ensures hemispheric balance (40% N, 40% S, 20% Equatorial)

#### Processors
- **ft8_decoder.py**: Decodes FT8 signals, extracts propagation characteristics
- **wspr_decoder.py**: Decodes WSPR beacons, preserves grid squares
- **qrn_analyzer.py**: Analyzes atmospheric noise, detects quiet periods
- **multichannel_qrn.py**: Extracts 9x overlapping 2.5kHz QRN channels
- **anonymizer.py**: One-way hashing of callsigns (preserves grid squares)
- **station_fingerprint.py**: Builds equipment signatures from signal characteristics
- **power_estimator.py**: SNR-distance triangulation for TX power estimation
- **statistical_power.py**: Bayesian inference using amateur radio power distributions
- **reciprocal_power.py**: Power asymmetry detection from bidirectional QSOs
- **pa_compression.py**: PA overdrive detection from harmonic analysis
- **multiband_power.py**: Cross-band power correlation for consistency
- **wspr_calibration.py**: Calibrates FT8 power estimates using WSPR ground truth
- **temporal_power.py**: Detects time-of-day and contest power patterns

#### Storage
- **compression.py**: FLAC encoding (45-55% size reduction)
- **tigris_storage.py**: Tigris S3 client for 35-75TB archive
- **file_manager.py**: Local buffering and lifecycle management
- **metadata_db.py**: PostgreSQL interface for recording metadata

#### Validators
- **quality_check.py**: Signal quality validation (keeps all data for complete archive)
- **coverage_check.py**: Ensures balanced collection across bands and times
- **qa_sampler.py**: 1% hot storage sampling for quality assurance
- **geographic_diversity.py**: Simpson's index, hemispheric balance, continental coverage
- **station_anomaly.py**: Detects unusual station behavior

#### Analytics
- **station_patterns.py**: Time-of-day and band preference analysis
- **station_aggregator.py**: Privacy-safe k-anonymity aggregation
- **rarity_scoring.py**: Prioritizes rare propagation events

#### Embeddings
- **station_aware.py**: Generates 128-D embeddings incorporating equipment fingerprints

## Data Flow

```
┌─────────────────┐
│ KiwiSDR/WebSDR  │ (50-100 global receivers)
│   Network       │
└────────┬────────┘
         │ 12 kHz IQ streams
         ▼
┌─────────────────┐
│    Scheduler    │ (Coordinates collection)
└────────┬────────┘
         │ Assigns SDRs to workers
         ▼
┌─────────────────┐
│  Worker Pool    │ (2-10 workers, auto-scaling)
│   - Record IQ   │
│   - Extract FT8 │
│   - Anonymize   │
└────────┬────────┘
         │ FLAC compressed + metadata
         ▼
┌─────────────────┐
│  Tigris S3      │ (35-75TB archive)
│  + PostgreSQL   │ (Metadata index)
└─────────────────┘
```

## Development Workflow

### Setting Up Local Environment

```bash
# Prerequisites
# - Docker & Docker Compose
# - Python 3.11+
# - Node.js 20+ (for dashboard)

# 1. Clone and setup
cd modules/data
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt

# 2. Start infrastructure
docker-compose up -d postgres keydb adminer redis-commander

# 3. Run migrations
python -m modules.data.migrations.run

# 4. Start services
python -m modules.data.src.collectors.scheduler &
python -m modules.data.src.collectors.worker &
python -m modules.data.src.dashboard.geographic_api &
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/unit/ -v                    # Unit tests
pytest tests/integration/ -v             # Integration tests
pytest tests/contract/ -v                # API contract tests

# Run with coverage
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html

# Run linting
ruff check src/
```

### Database Migrations

```bash
# Create a new migration
cd migrations/
# Edit SQL file with schema changes

# Apply migrations
python -m modules.data.migrations.run

# Check current schema
docker-compose exec postgres psql -U postgres -d cascade_data -c "\dt"
```

### Adding Test Data

```bash
# Add geographic diversity test samples
curl -X POST http://localhost:8000/api/diversity/add-sample \
  -H "Content-Type: application/json" \
  -d '{
    "grid_square": "FN42",
    "hours": 10,
    "latitude_band": "temperate",
    "hemisphere": "north",
    "is_ocean_path": false
  }'

# View diversity metrics
curl http://localhost:8000/api/diversity/metrics | jq
```

## Configuration

### Database Requirements

The CASCADE Data Module requires PostgreSQL 15+ with significant storage:

| Phase | Database Size | RAM Required | Recommended Config |
|-------|--------------|--------------|-------------------|
| **Development** | 5-20 GB | 2GB | Single node, shared-cpu-2x |
| **Production (6mo)** | ~70 GB | 8GB | 3-node HA, shared-cpu-4x |
| **Production (18mo)** | ~250 GB | 8-16GB | 3-node HA, shared-cpu-4x or performance-1x |
| **Long-term** | 500GB+ | 16GB+ | 3-node HA, performance-2x |

**Storage Growth**: ~14 GB/month during active collection
- **recording_sessions**: 250K records, ~250 MB
- **propagation_records**: 600M FT8/WSPR decodes, ~120 GB
- **qrn_samples**: 90M samples, ~13.5 GB
- **station_fingerprints**: 50K stations, ~100 MB
- **Other metadata**: ~1 GB

**See [DEPLOYMENT.md](./DEPLOYMENT.md)** for Postgres setup, HA configuration, and cost analysis.

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://postgres:postgres@localhost:5432/cascade_data` |
| `REDIS_URL` | Redis/KeyDB connection string | `redis://localhost:6379/0` |
| `TIGRIS_ACCESS_KEY` | Tigris S3 access key | Required |
| `TIGRIS_SECRET_KEY` | Tigris S3 secret key | Required |
| `TIGRIS_BUCKET` | S3 bucket name | `cascade-iq-data` |
| `NOAA_API_KEY` | NOAA space weather API key | Optional |
| `DIVERSITY_MONITORING` | Enable geographic diversity checks | `enabled` |
| `SCARCE_REGION_SLOTS` | Reserved slots for underrepresented regions | `0.2` (20%) |
| `PREFER_SCARCE_REGIONS` | Prioritize southern hemisphere | `true` |
| `MIN_SDRS` | Minimum concurrent SDRs | `6` |
| `MAX_SDRS` | Maximum SDRs during events | `50` |

### Configuration Files

- **fly.toml**: Fly.io deployment configuration
- **docker-compose.yml**: Local development services
- **pyproject.toml**: Python project metadata and linting rules
- **requirements.txt**: Python dependencies

## Key Features

### Geographic Diversity Monitoring

The module includes a real-time dashboard tracking:
- **Simpson's Diversity Index**: Measures distribution uniformity (target: >0.7)
- **Hemispheric Balance**: North/South ratio (target: 0.8-1.2)
- **Continental Coverage**: Tracks all 7 continents
- **Latitude Band Quotas**: Ensures 20% minimum per band (Arctic, Temperate, Tropical, Antarctic)

### Station Fingerprinting

Non-invasive characterization of transmitter equipment:
- Phase noise analysis from FT8 symbol centers
- Frequency drift tracking across observations
- PA linearity estimation from spectral regrowth
- Equipment-propagation separation using multi-diversity
- Temporal pattern detection (time-of-day, contest mode)

### Advanced Power Estimation

Sophisticated TX power estimation without explicit reports:
- **SNR-Distance Triangulation**: Multi-receiver path loss calculations
- **Statistical Analysis**: Bayesian inference with amateur radio priors
- **Reciprocal Path Detection**: Power asymmetry from bidirectional QSOs
- **PA Compression Analysis**: Overdrive detection from harmonics/IMD
- **Multi-Band Correlation**: Cross-band consistency checking
- **WSPR Calibration**: Ground truth from explicit WSPR power reports
- **Temporal Patterns**: Contest vs casual power level detection

### Privacy-Preserving Design

- **Callsign Anonymization**: One-way SHA-256 hashing with salt
- **Grid Square Preservation**: Maintains geographic context for propagation analysis
- **K-Anonymity Aggregation**: Statistical summaries with minimum group sizes
- **No PII Storage**: Only technical signal characteristics retained

## Testing Strategy

### Test-Driven Development (TDD)

The project follows strict TDD:
1. **Contract Tests First**: Define API behavior (Phase 3.2)
2. **Integration Tests**: Test full pipelines (Phase 3.2)
3. **Implementation**: Write code to pass tests (Phase 3.3)
4. **Unit Tests**: Add comprehensive coverage (Phase 3.5)

### Test Coverage Requirements

- **Minimum**: 80% line coverage
- **Critical paths**: 100% coverage (collectors, anonymization, storage)
- **Contract tests**: All API endpoints
- **Integration tests**: End-to-end collection scenarios

## Common Development Tasks

### Adding a New Processor

```python
# 1. Create processor file
# src/processors/my_processor.py

from typing import Dict, Any

class MyProcessor:
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        # Implementation
        return processed_data

# 2. Write tests first
# tests/unit/test_my_processor.py

def test_my_processor():
    processor = MyProcessor()
    result = processor.process({"input": "data"})
    assert result["output"] == "expected"

# 3. Add to pipeline
# src/collectors/worker.py
from processors.my_processor import MyProcessor
```

### Adding a New Database Model

```python
# 1. Define model
# src/models/my_model.py

from sqlalchemy import Column, Integer, String
from .base import Base

class MyModel(Base):
    __tablename__ = "my_table"

    id = Column(Integer, primary_key=True)
    name = Column(String(255))

# 2. Create migration
# migrations/003_add_my_table.sql

CREATE TABLE my_table (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL
);

# 3. Apply migration
python -m modules.data.migrations.run
```

### Adding a New API Endpoint

```python
# src/api/my_endpoints.py

from fastapi import APIRouter

router = APIRouter()

@router.get("/my-endpoint")
async def my_endpoint():
    return {"message": "Hello"}

# Register in main app
# src/api/main.py
from .my_endpoints import router as my_router
app.include_router(my_router, prefix="/api")
```

## Deployment

See [DEPLOYMENT.md](./DEPLOYMENT.md) for detailed deployment instructions.

**Quick Deploy to Fly.io**:
```bash
./deploy.sh
```

## Monitoring & Debugging

### Logs

```bash
# Docker Compose
docker-compose logs -f scheduler
docker-compose logs -f worker

# Fly.io
fly logs -a cascade-kiwi-collector

# Filter by process
fly logs -a cascade-kiwi-collector | grep "worker"
```

### Metrics

- **Prometheus**: http://localhost:9090/metrics
- **Fly Dashboard**: https://fly.io/dashboard/cascade-kiwi-collector
- **Geographic Diversity**: http://localhost:3000

### Common Issues

**Workers not connecting to SDRs**:
- Check SDR availability: `curl http://kiwisdr.example.com:8073`
- Verify usage limits haven't been exceeded
- Check network connectivity

**Database connection errors**:
- Ensure Postgres is running: `docker-compose ps postgres`
- Check DATABASE_URL environment variable
- Run migrations: `python -m modules.data.migrations.run`

**Storage issues**:
- Verify Tigris credentials: `echo $TIGRIS_ACCESS_KEY`
- Check bucket exists and is accessible
- Monitor disk space: `df -h`

## Contributing

1. **Follow TDD**: Write tests before implementation
2. **Run linting**: `ruff check src/` before committing
3. **Update docs**: Keep README and DEPLOYMENT.md current
4. **Privacy first**: Never log/store PII or unencrypted callsigns
5. **Test coverage**: Maintain >80% coverage

## Project Status

**Completed** (136/216 tasks - 63%):
- ✅ Core collection infrastructure
- ✅ FT8/WSPR decoding
- ✅ Station fingerprinting
- ✅ Geographic diversity monitoring
- ✅ Advanced power estimation
- ✅ Privacy-preserving anonymization

**In Progress**:
- ⏳ Dashboard waterfall viewer
- ⏳ Event-driven scaling (solar events, contests)
- ⏳ Comprehensive unit test suite

## License

Copyright (c) 2024-2025 CASCADE Project. All rights reserved.

## Support

- **Issues**: https://github.com/cascade/data-collector/issues
- **Documentation**: `/docs` directory
- **Spec**: `/specs/001-kiwi-data-collector/`