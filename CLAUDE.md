# CASCADE KiwiSDR Data Collector - Claude Development Guide

## Project Context

CASCADE (Cognitive Adaptive Spectrum Coordination And Distributed Efficiency) is building a neural network-based adaptive HF radio system using **128-pattern chaos architecture** with **kernel-driven emergent coordination**, achieving **78-85% Shannon efficiency** and supporting **1,024 total users** (45 active).

You're working on the **Data Module**, which collects 24,000-36,000 hours of real-world HF radio data for training the neural network model. This data captures real atmospheric noise (QRN) and ionospheric propagation characteristics that synthetic models cannot replicate.

**Architecture Status (Oct 2025):** ✅ Complete - 128 patterns, 135-tone grid (300-3000 Hz), RS(32,20) structure, kernel lifecycle protocol finalized. See [docs/architecture.md](docs/architecture.md) for executive summary.

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

## CASCADE Architecture (Final - Dual-Layer)

**Pattern System (2-FSK, All λ=0):**
- 128 patterns (48 beacon + 80 message, 7-bit pattern ID)
- 2-FSK architecture: Each pattern uses 2 adjacent tones (tone indices 0-1)
- 135-tone reference grid (300-3000 Hz, 20 Hz spacing, standard 2.7 kHz SSB channel)
- Symbol rate: 200 symbols/second (5ms per symbol)
- Pattern duration: 160ms (32 symbols × 5ms)
- All patterns λ=0 (BPSK baseline for pattern skeleton)
- RS(32,20) equivalent: Pattern recognition with 37.5% erasure tolerance (QR code-like)
- Storage: 38 KB, Generation: 48-60 hours (simplified, frequency-only optimization)

**Dual-Layer Encoding:**
- **Layer 1 (Pattern ID)**: Time×Frequency hopping sequence identifies pattern (7 bits)
  - Orthogonality: -37.5 dB in Time×Frequency space (no IQ needed for separation)
  - All λ=0: BPSK skeleton, maximum robustness
- **Layer 2 (Data Payload)**: Adaptive IQ modulation encodes user data (8-32 bits)
  - BPSK (SNR<0): 8 bits, 94 bps per pattern
  - QPSK (SNR 0-10): 16 bits, 144 bps per pattern
  - 8-PSK (SNR 10-20): 24 bits, 194 bps per pattern
  - 16-APSK (SNR>20): 32 bits, 244 bps per pattern (4+12 constellation, better PA efficiency)
  - Differential encoding: Immune to ±0.1-10 Hz frequency drift (no GPS required)
  - Tone revisit redundancy: 4× average (natural error correction)

**Kernel Architecture (4 kernels × 28 bytes = 112 bytes per beacon):**
- Three RX kernels: Best ways for others to reach this station (decoder-generated from IQ)
- One TX kernel: Current transmission state for collision avoidance (protocol-generated)
- Each kernel (28 bytes):
  - Pattern range: Start ID (7b) + Count (3b) for 1-8 consecutive patterns
  - Channel: Frequency pair (7b) + Modulation (3b)
  - Version: Protocol (2b) + Model (2b)
  - Embedding: 48 dims × 4-bit quantization (learned optimization hints)
- Multi-pattern support: Stations transmit 1-8 patterns based on RX capability and conditions
- Distributed coordination: TX kernels provide anti-collision, RX kernels guide optimization

**Performance (Equipment-Adaptive):**
- Shannon efficiency: 55-70% channel (adaptive modulation) × 78-85% coordination = 45-60% system
- Active users: 40-45 simultaneous (adaptive pattern allocation)
  - QRP stations: 20-30 users × 1 pattern each = 20-30 patterns
  - Medium stations: 8-10 users × 4 patterns each = 32-40 patterns
  - Premium stations: 2-3 users × 8 patterns each = 16-24 patterns
  - Total: ~80 message patterns efficiently shared
- Total network capacity: 1,024 users (via frequency/time/geographic reuse)
- Throughput per user:
  - QRP/QMX (5W, 200 sym/s): 94 bps (1 pattern, BPSK)
  - Modern (50W, 200 sym/s): 575 bps (4 patterns, QPSK)
  - Premium (100W, 300 sym/s): 975-1,950 bps (4-8 patterns, 8-PSK/16-APSK)
- Hardware: Raspberry Pi 4 + Coral Edge TPU ($110, 15W, portable)

**Training Data Needs:**
- 24-36K hours real HF recordings (QRN + propagation)
- Multi-modulation training: BPSK through 16-APSK
- Encoder mutations: Continuous signal optimization
- Decoder dual-role: Pattern recognition + kernel generation

## Recent Changes
- **Oct 2025**: Architecture finalized - Dual-layer 2-FSK with adaptive modulation and kernel coordination
  - Pattern generation: 2-FSK (tone indices 0-1), all λ=0, 135-tone grid @ 20 Hz
  - Dual-layer: Pattern ID (7b) + Adaptive data (8-32b) = 15-39 bits/pattern total
  - Kernel: 28 bytes (17b discrete + 8b version + 24B embedding), top-3 candidates
  - Throughput: 75-450 bps/pattern @ 200 sym/s (BPSK to 16-APSK), equipment-adaptive
  - Hardware: RPi4 + Coral TPU ($110, 15W) for full 45-user network decode
- **V1 MVP Strategy**: 24-36K hours KiwiSDR data for initial model training
- Target: 133 cooperating KiwiSDR owners @ 60 min/day
- Storage: 4-7TB for V1 data collection
- Geographic: V1 bias (65% N, 15% S, 20% Eq) corrected via post-V1 telemetry
- Data feeds model training (see [docs/training/README.md](docs/training/README.md))

## See Also
- **[CASCADE Architecture](docs/architecture.md)** - Executive summary of 128-pattern chaos system
- **[Training Strategy](docs/training/README.md)** - How collected data is used for model training
- **[Telemetry Research](docs/telemetry_research.md)** - Post-deployment continuous improvement

---
*Last updated: 2025-10-04*