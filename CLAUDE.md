# CASCADE KiwiSDR Data Collector - Claude Development Guide

## Project Context
CASCADE (Cognitive Adaptive Spectrum Coordination And Distributed Efficiency) is building a neural network-based adaptive radio system. You're working on the Data Module, which collects 24,000-36,000 hours of real-world HF radio data (4,000-6,000 hours per band) from the global KiwiSDR network over 6 months for V1 launch. Post-V1, telemetry from deployed systems will improve performance in underserved regions over 12-18 months.

## Current Feature: KiwiSDR Data Collector
- **Branch**: 001-kiwi-data-collector
- **Stage**: Implementation starting
- **Goal**: Collect QRN and FT8/WSPR propagation data for model training

## Technical Stack
- **Language**: Python 3.11
- **Key Libraries**: kiwiclient, numpy, scipy, pandas, psycopg2
- **Storage**: 4-7TB Tigris S3 with FLAC compression (45-55% size reduction)
- **Database**: PostgreSQL for metadata

## Module Structure
```
modules/data/
├── src/
│   ├── collectors/     # KiwiSDR connections
│   ├── processors/     # FT8/WSPR decoders
│   ├── validators/     # Quality checks
│   └── storage/        # File management
└── tests/             # Pytest test suite
```

## Key Implementation Details

### KiwiSDR Integration
- Use kiwiclient library for connections
- 12 kHz IQ mode at 12 kHz sample rate
- Coordinate with 133 cooperating KiwiSDR owners
- Handle 90-minute daily usage limits per SDR

### Data Processing
- Real-time FT8 extraction during collection
- Anonymize callsigns via one-way hashing
- FLAC compression for 45-55% size reduction
- Keep all data regardless of quality

### Collection Strategy (V1 MVP)
- 6 stations continuous (one per HF band)
- Scale to 20+ stations during solar events
- Target 133-200 hours/day collection rate
- 6-month timeline for 24,000-36,000 total hours
- Geographic focus: NA/EU/Japan (90-95% V1 performance)
- Post-V1: Telemetry fills underserved regions over 12-18 months

## Constitutional Requirements
1. **Data-First**: This is the first module - no dependencies
2. **Test-Driven**: Write failing tests before implementation
3. **Real Data**: No synthetic data - only real KiwiSDR recordings
4. **Privacy**: Anonymize all callsigns and locations
5. **Reproducible**: Deterministic processing, versioned datasets

## Testing Requirements
```python
# Test structure for TDD
def test_kiwi_connection():
    """Test connection to KiwiSDR"""
    assert False  # Write this first

def test_ft8_extraction():
    """Test FT8 signal extraction"""
    assert False  # Then implement
```

## Common Commands
```bash
# Run tests
pytest modules/data/tests/ -v

# Start collection
python -m cascade_collector start --bands 20m,40m

# Check status
python -m cascade_collector status

# Validate recording
python -m cascade_validator check recording.wav
```

## V1 MVP Implementation Plan

### Phase 1: Update Configuration Constants (Week 1)
**Goal**: Update hardcoded values throughout codebase to match V1 targets

1. **Collection Targets** (`src/config/*.py`):
   - `TARGET_TOTAL_HOURS = 24000`  # Minimum baseline
   - `TARGET_HOURS_PER_BAND = 4000`  # 6 bands
   - `COLLECTION_DURATION_MONTHS = 6`
   - `TARGET_DAILY_HOURS = 133`  # Minimum (24000 / 180 days)

2. **SDR Pool Configuration** (`src/collectors/sdr_manager.py`):
   - `MIN_KIWISDR_OWNERS = 133`  # Cooperating owners needed
   - `KIWISDR_DAILY_LIMIT_HOURS = 1.0`  # 60 min average
   - `CONCURRENT_CONNECTIONS_BASELINE = 15`  # Down from 50-100
   - Remove WebSDR support (KiwiSDR-only for V1)

3. **Geographic Distribution** (`src/collectors/geographic_quotas.py`):
   - `NORTHERN_HEMISPHERE_TARGET = 0.65`  # Accept bias for V1
   - `SOUTHERN_HEMISPHERE_TARGET = 0.15`  # Minimal for V1
   - `EQUATORIAL_TARGET = 0.20`
   - Add V1 relaxed quotas vs comprehensive targets

4. **Storage Sizing** (`src/storage/tigris_storage.py`, `src/config/storage_config.py`):
   - `EXPECTED_TOTAL_STORAGE_TB = 6`  # Mid-range 4-7TB
   - `MONTHLY_STORAGE_GROWTH_GB = 1000`  # ~1TB/month
   - Update Tigris bucket lifecycle policies

5. **Database Sizing** (migrations, deployment configs):
   - Update PostgreSQL size estimates: 50GB for 6 months
   - Adjust connection pool sizes for lower load
   - Update fly.toml database configuration

### Phase 2: Simplify Architecture (Week 2)
**Goal**: Remove complexity not needed for V1 MVP

1. **Remove WebSDR Support**:
   - Delete `src/collectors/websdr_client.py`
   - Remove hybrid selection logic from `hybrid_sdr_selector.py`
   - Update tests to remove WebSDR scenarios

2. **Simplify Event Scaling**:
   - Remove aggressive 100-200 SDR scaling logic
   - Keep simple 2x multiplier for K≥5 storms
   - V1 focus: baseline collection, not event bursts

3. **Reduce Worker Scaling**:
   - Fly.io autoscaling: 2-5 workers (down from 2-10)
   - Lower memory/CPU requirements per worker
   - Simpler coordination (fewer workers = less Redis overhead)

4. **Simplify Geographic Diversity**:
   - Remove strict 40/40/20 enforcement
   - Add "relaxed mode" flag for V1 (65/15/20 acceptable)
   - Keep tracking but don't block on quotas

### Phase 3: Update Scheduling Logic (Week 3)
**Goal**: Optimize scheduler for V1 target of 133-200 hrs/day

1. **Scheduler Configuration** (`src/collectors/scheduler.py`):
   - Target: 133 hrs/day baseline (vs 200-500 previously)
   - Simplified prioritization (no complex seasonal weighting)
   - KiwiSDR-only rotation logic

2. **Queue Management** (`src/collectors/queue_manager.py`):
   - Lower queue depth (fewer concurrent workers)
   - Simpler retry logic (no WebSDR fallback)
   - Update Redis key TTLs for 6-month collection

3. **SDR Selection** (`src/collectors/sdr_manager.py`):
   - Prioritize reliability over diversity for V1
   - Accept northern hemisphere concentration
   - Track participation: aim for 133 consistent owners

### Phase 4: Update Monitoring & Dashboards (Week 4)
**Goal**: Dashboard shows progress toward V1 targets

1. **Progress Tracking**:
   - Show: "24,000 / 36,000 hours collected (67%)"
   - Per-band progress: "4,200 / 6,000 hours on 20m (70%)"
   - Days remaining: "90 days left in 6-month collection"

2. **Geographic Dashboard** (`src/dashboard/geographic_api.py`):
   - Update targets: 65/15/20 instead of 40/40/20
   - Show "V1 Relaxed Mode" indicator
   - Color coding: green if meeting V1 targets

3. **Cost Dashboard**:
   - Track actual spend vs $3,000 budget
   - Projected total cost for 6 months
   - Storage costs (should stay under $500 total)

4. **Participant Tracking**:
   - Show: "87 / 133 cooperating owners active"
   - Top contributors (anonymized by region)
   - Daily average per-owner contribution

### Phase 5: Update Documentation & Alerts (Week 5)
**Goal**: System communicates V1 strategy clearly

1. **Update Email Notifications** (`src/notifications/gmail_notifier.py`):
   - Subject: "CASCADE V1: Daily Status"
   - Content references V1 targets (not comprehensive)
   - Alerts if falling behind V1 pace

2. **API Response Updates** (`src/api/*.py`):
   - Return V1 targets in `/metrics` endpoint
   - Add `/v1-status` endpoint showing progress
   - Document V1 vs comprehensive differences

3. **Logging & Telemetry**:
   - Log messages reference "V1 collection"
   - Track "days to V1 target" metric
   - Warning if pace < 133 hrs/day for 7 days

### Phase 6: Testing & Validation (Week 6)
**Goal**: Ensure system works for V1 targets

1. **Update Unit Tests**:
   - Test with 133-200 hrs/day collection rate
   - Test 65/15/20 geographic distribution
   - Test 6-month duration calculations

2. **Update Integration Tests**:
   - Simulate 6-month collection in fast-forward
   - Verify storage stays under 7TB
   - Test with KiwiSDR-only pool

3. **Contract Tests**:
   - API returns V1 targets correctly
   - Dashboard displays V1 progress
   - Metrics align with 24-36K hour goals

### Phase 7: Deployment Updates (Week 7)
**Goal**: Deploy V1-optimized configuration

1. **Update fly.toml**:
   - Lower resource allocations (smaller DB, fewer workers)
   - Update environment variables for V1 targets
   - Cost-optimized machine types

2. **Database Migrations**:
   - Add `collection_strategy` column: 'v1_mvp' vs 'comprehensive'
   - Add V1 milestone tracking table
   - Update indexes for lower volume

3. **Deployment Scripts**:
   - `deploy.sh` checks for V1 configuration
   - Health checks validate V1 targets
   - Monitoring alerts tuned for V1 scale

## Current Tasks
See `/specs/001-kiwi-data-collector/tasks.md` for detailed implementation tasks.

## Known Issues & Solutions
- **KiwiSDR timeout**: Implement exponential backoff
- **Storage overflow**: Auto-prune old processed files
- **CPU bottleneck**: Distribute processing across cores

## Recent Changes
- **V1 MVP Strategy**: Reduced to 6-month, 24-36K hours for faster launch
- Target: 133 cooperating KiwiSDR owners @ 60 min/day
- Storage: 4-7TB (down from 35-75TB)
- Geographic: Accept V1 bias (65% N, 15% S, 20% Eq), telemetry fills gaps
- Added distributed Fly.io architecture with Redis/KeyDB coordination
- Defined data model and API contracts

---
*Last updated: 2025-09-29 by /plan command*