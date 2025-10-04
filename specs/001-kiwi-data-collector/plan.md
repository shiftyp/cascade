# Implementation Plan: KiwiSDR Data Collector

**Branch**: `001-kiwi-data-collector` | **Date**: 2025-09-29 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-kiwi-data-collector/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path ✓
2. Fill Technical Context ✓
3. Fill the Constitution Check section ✓
4. Evaluate Constitution Check section ✓
5. Execute Phase 0 → research.md ✓
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md ✓
7. Re-evaluate Constitution Check section ✓
8. Plan Phase 2 → Describe task generation approach ✓
9. STOP - Ready for /tasks command
```

## Summary
Collect 200,000-250,000 hours of real-world atmospheric noise and propagation data from global KiwiSDR and WebSDR receivers over 18 months (minimum 10,000 per HF band baseline). Implement hybrid collection strategy leveraging WebSDR's higher capacity and KiwiSDR's geographic diversity. Process FT8/WSPR signals to extract propagation characteristics while preserving privacy through anonymization. Store compressed IQ data and extracted features in expandable 40-50TB archive for comprehensive CASCADE neural network training.

## Technical Context
**Language/Version**: Python 3.11
**Primary Dependencies**: kiwiclient, websdr-client, numpy, scipy, pandas, pyFT8, wsjtx-lib, redis-py
**Storage**: File-based (FLAC compressed) with PostgreSQL metadata database, Redis/KeyDB for coordination
**Testing**: pytest, pytest-mock for KiwiSDR and WebSDR mocking
**Target Platform**: Fly.io distributed (2-10 machines), Ubuntu 22.04 LTS base
**Project Type**: Data Module in CASCADE monorepo
**Performance Goals**: Process 300-400 hours/day across 50-100 concurrent streams from pool of 800-1100 SDRs
**Constraints**: Respect KiwiSDR (600-800 available) 30-90 minute daily limits and WebSDR (200-300 available) institutional policies, anonymize all PII
**Scale/Scope**: 200,000-250,000 hours collection, 40-50TB storage, 18-month timeline, hybrid KiwiSDR/WebSDR strategy
**Deployment**: Fly.io with auto-scaling, Redis/KeyDB for distributed coordination

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Data-First Development**: This IS the data module - first in dependency chain
- [x] **Monorepo Module Architecture**: Isolated to modules/data/ directory
- [x] **Clean Separation**: N/A - no protocol/model separation in data collection
- [x] **Test-Driven Development**: Tests planned for all components
- [x] **Real-World Data Priority**: Exclusively real KiwiSDR/FT8/WSPR data
- [x] **Privacy-Preserving**: Callsign anonymization, no message content stored
- [x] **Reproducible Research**: Deterministic processing, versioned datasets

## Project Structure

### Documentation (this feature)
```
specs/001-kiwi-data-collector/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root - monorepo structure)
```
# CASCADE Monorepo Structure

modules/data/
├── src/
│   ├── collectors/
│   │   ├── kiwi_client.py      # KiwiSDR connection manager
│   │   ├── recorder.py         # IQ recording orchestrator
│   │   ├── scheduler.py        # Collection scheduling (master)
│   │   ├── worker.py           # Distributed worker process
│   │   └── queue_manager.py    # Redis queue coordination
│   ├── processors/
│   │   ├── ft8_decoder.py      # FT8 signal extraction
│   │   ├── wspr_decoder.py     # WSPR signal extraction
│   │   ├── qrn_analyzer.py     # Noise characterization
│   │   └── anonymizer.py       # PII removal
│   ├── validators/
│   │   ├── quality_check.py    # Data quality validation
│   │   └── coverage_check.py   # Geographic/temporal coverage
│   └── storage/
│       ├── file_manager.py     # FLAC file handling
│       ├── metadata_db.py      # PostgreSQL interface
│       └── compression.py      # FLAC compression utilities
└── tests/
    ├── unit/
    │   ├── test_collectors/
    │   ├── test_processors/
    │   └── test_storage/
    └── integration/
        ├── test_collection_pipeline.py
        └── test_kiwisdr_rotation.py
```

**Structure Decision**: Data Module only - this is the foundational CASCADE component with no dependencies on other modules.

## Fly.io Deployment Architecture

### Machine Configuration
```toml
# fly.toml
app = "cascade-kiwi-collector"
primary_region = "iad"

[build]
  dockerfile = "Dockerfile"

# Scheduler Machine (always 1 instance)
[[vm]]
  memory = "2gb"
  cpus = 2

[processes]
  scheduler = "python -m cascade_collector.scheduler"
  worker = "python -m cascade_collector.worker"

# Auto-scaling for worker machines
[[services]]
  processes = ["worker"]

  [[services.autoscale]]
    min_machines = 2
    max_machines = 10

    [[services.autoscale.policy]]
      metric = "connections"  # SDR connections per worker
      threshold = 5
      cooldown = 60

    [[services.autoscale.policy]]
      metric = "cpu"
      threshold = 70
      cooldown = 120

# Redis/KeyDB Instance
[redis]
  image = "eqalpha/keydb"
  memory = "1gb"

# Internal networking
[network]
  services = ["redis", "postgres", "tigris"]
```

### Distributed Coordination Flow
```
1. Scheduler (1 machine):
   - Monitors space weather APIs
   - Calculates propagation conditions
   - Pushes SDR assignments to Redis queue
   - Manages global collection schedule

2. Workers (2-10 machines):
   - Pull assignments from Redis queue
   - Claim SDRs with distributed locks
   - Stream IQ data → FLAC compression
   - Upload to Tigris S3
   - Report health to Redis

3. Redis/KeyDB:
   - Work queue for SDR assignments
   - Distributed locks for SDR claims
   - Health/status tracking
   - Event pub/sub for scaling
```

## Phase 0: Outline & Research

### Research Priorities
1. **KiwiSDR Network Analysis**
   - Public receiver availability and geographic distribution
   - Usage limits and rotation strategies
   - API capabilities and limitations

2. **FT8/WSPR Processing**
   - Optimal recording windows (12 kHz bandwidth with specific center frequencies):
     - 80m: 3576 kHz (WSPR + FT8 + quiet spectrum)
     - 40m: 7080 kHz (FT8 + quiet digital sub-band)
     - 20m: 14080 kHz (FT8 + quiet zone)
     - 15m: 21080 kHz (FT8 + quiet zone)
     - 10m: 28080 kHz (FT8 + quiet zone)
     - 6m: 50303 kHz (WSPR + quiet zone)
   - Propagation mutation extraction techniques
   - Real-time vs batch processing tradeoffs

3. **Storage Optimization**
   - FLAC compression ratios for IQ data
   - Tiered storage strategies
   - Cloud vs local storage costs

**Output**: research.md with technical decisions and rationale

## Phase 1: Design & Contracts

### Data Model (`data-model.md`)
Primary entities (8):
- **RecordingSession**: IQ recordings with metadata and correlation IDs
- **KiwiSDRSource**: Receiver registry with usage tracking
- **QRNSample**: Noise characterization with quiet period markers
- **PropagationRecord**: FT8/WSPR with propagation mode labels and grid squares
- **SpaceWeatherData**: NOAA data including X-ray classification
- **CollectionSchedule**: Automated recording plans
- **QASample**: Quality assurance samples with review status
- **NotificationConfig**: Alert configuration

Supporting tables (2):
- **CollectionAlerts**: SDR availability and error notifications
- **NotificationTemplates**: Message templates for alerts

### API Contracts (`contracts/`)
- `collector_api.yaml`: KiwiSDR connection interface
- `processor_api.yaml`: Signal extraction interface
# (storage and scheduling are internal services, not API contracts)

### Quickstart Guide (`quickstart.md`)
- Environment setup (Python 3.11, dependencies)
- KiwiSDR list configuration
- Running first collection
- Verifying data quality

### Agent File (`CLAUDE.md`)
- Project context and goals
- Module structure and conventions
- Testing requirements
- Common issues and solutions

## Phase 2: Task Planning Approach

### Task Generation Strategy
1. **Setup Tasks**: Environment, dependencies, configuration
2. **Test-First Tasks**: Write failing tests for each component
3. **Implementation Tasks**: Make tests pass incrementally
4. **Integration Tasks**: Connect components into pipeline
5. **Validation Tasks**: Verify 150,000+ hour collection capability
6. **Event Scaling Tasks**: Implement propagation event detection and SDR scaling
7. **Training Support Tasks**: Correlation preservation, mode detection, multi-channel extraction
8. **Resilience Tasks**: SDR degradation handling, health monitoring, minimum operation mode
9. **QA Tasks**: Quality sampling, reporting, and quarantine procedures
10. **Distributed Infrastructure Tasks**: Redis queue setup, worker coordination, Fly.io auto-scaling

### Task Ordering
- TDD order: Tests before implementation
- Dependency order: Storage → Processors → Collectors → Scheduler
- Parallel execution for independent components

**Estimated Output**: 113 tasks organized by phase and component (including distributed coordination and QA visualization)

## Complexity Tracking
No violations - this is appropriately scoped for the Data Module.

## Progress Tracking

**Phase Status**:
- [x] Phase 0: Research complete
- [x] Phase 1: Design complete
- [x] Phase 2: Task planning complete
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none)

---
*Based on Constitution v1.0.0 - See `.specify/memory/constitution.md`*